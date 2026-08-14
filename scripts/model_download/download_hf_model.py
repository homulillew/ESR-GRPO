"""下载并校验一个 Hugging Face safetensors 模型，支持断点续传和固定 commit。"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from esr_grpo.model_download import load_manifest, verify_model_directory


DEFAULT_ALLOW = (
    "*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja", "*.py", "*.md",
)
DEFAULT_IGNORE = ("*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "flax_model*", "original/*")
PROXY_VARIABLES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)


@contextmanager
def temporary_environment(changes: dict[str, str | None]):
    before = {key: os.environ.get(key) for key in changes}
    try:
        for key, value in changes.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class DownloadLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "DownloadLock":
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise SystemExit(f"download lock exists: {self.path}; another download may be running") from exc
        os.write(self.fd, f"pid={os.getpid()}\n".encode())
        return self

    def __exit__(self, *_: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def selected(path: str, allow: Iterable[str], ignore: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in allow) and not any(
        fnmatch.fnmatch(path, pattern) for pattern in ignore
    )


def sibling_metadata(sibling: Any) -> dict[str, Any]:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        sha256 = lfs.get("sha256")
    else:
        sha256 = getattr(lfs, "sha256", None)
    return {
        "path": sibling.rfilename,
        "size": getattr(sibling, "size", None),
        "sha256": sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--endpoint", help="例如 https://hf-mirror.com；默认使用 Hugging Face 官方端点")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--allow-pattern", action="append")
    parser.add_argument("--ignore-pattern", action="append")
    parser.add_argument("--reserve-gib", type=float, default=5.0)
    parser.add_argument("--skip-disk-check", action="store_true")
    parser.add_argument("--ignore-environment-proxy", action="store_true")
    parser.add_argument("--verify-lfs-sha256", action="store_true", help="完整读取约 60 GB 权重进行哈希校验")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只显示参数，不访问网络或磁盘")
    args = parser.parse_args()

    allow = tuple(args.allow_pattern or DEFAULT_ALLOW)
    ignore = tuple(args.ignore_pattern or DEFAULT_IGNORE)
    request = {
        "repo_id": args.repo_id,
        "output": str(args.output.resolve()),
        "revision": args.revision,
        "allow_patterns": allow,
        "ignore_patterns": ignore,
        "max_workers": args.max_workers,
        "endpoint": args.endpoint,
        "verify_only": args.verify_only,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **request}, ensure_ascii=False, indent=2))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "download_manifest.json"
    with DownloadLock(args.output / ".download.lock"):
        if args.verify_only:
            if not manifest_path.is_file():
                raise SystemExit(f"manifest does not exist: {manifest_path}")
            manifest = load_manifest(manifest_path)
            result = verify_model_directory(
                args.output,
                manifest.get("expected_files", []),
                verify_lfs_sha256=args.verify_lfs_sha256,
            )
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            if not result.valid:
                raise SystemExit(1)
            return

        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise SystemExit("install downloader dependencies: pip install -r requirements-download.txt") from exc

        token = os.environ.get(args.token_env) or None
        environment_changes = {key: None for key in PROXY_VARIABLES} if args.ignore_environment_proxy else {}
        if args.endpoint:
            environment_changes["HF_ENDPOINT"] = args.endpoint
        with temporary_environment(environment_changes):
            api = HfApi(endpoint=args.endpoint, token=token) if args.endpoint else HfApi(token=token)
            info = api.model_info(args.repo_id, revision=args.revision, files_metadata=True, token=token)
            commit = str(info.sha)
            expected = [
                sibling_metadata(item)
                for item in (info.siblings or ())
                if selected(item.rfilename, allow, ignore)
            ]
            known_total = sum(int(item["size"]) for item in expected if item["size"] is not None)
            remaining = sum(
                int(item["size"])
                for item in expected
                if item["size"] is not None
                and (not (args.output / item["path"]).is_file()
                     or (args.output / item["path"]).stat().st_size != int(item["size"]))
            )
            free = shutil.disk_usage(args.output).free
            required = int(remaining * 1.10 + args.reserve_gib * 2**30)
            plan = {
                **request,
                "resolved_revision": commit,
                "selected_files": len(expected),
                "known_total_bytes": known_total,
                "remaining_bytes": remaining,
                "free_bytes": free,
                "required_free_bytes": required,
            }
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            if not args.skip_disk_check and free < required:
                raise SystemExit(
                    f"insufficient disk space: free={free / 2**30:.1f} GiB, "
                    f"required={required / 2**30:.1f} GiB"
                )

            download_kwargs = dict(
                repo_id=args.repo_id,
                repo_type="model",
                revision=commit,
                local_dir=args.output,
                token=token,
                allow_patterns=list(allow),
                ignore_patterns=list(ignore),
                max_workers=args.max_workers,
            )
            if args.endpoint:
                download_kwargs["endpoint"] = args.endpoint
            snapshot_download(**download_kwargs)

        result = verify_model_directory(
            args.output, expected, verify_lfs_sha256=args.verify_lfs_sha256
        )
        manifest = {
            "format_version": 1,
            "repo_id": args.repo_id,
            "requested_revision": args.revision,
            "resolved_revision": commit,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "expected_files": expected,
            "verification": result.to_dict(),
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
        print(json.dumps({"manifest": str(manifest_path), **result.to_dict()}, ensure_ascii=False, indent=2))
        if not result.valid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
