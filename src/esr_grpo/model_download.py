"""Hugging Face 大模型下载器使用的清单、磁盘和本地分片校验。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    errors: tuple[str, ...]
    checked_files: int
    checked_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "checked_files": self.checked_files,
            "checked_bytes": self.checked_bytes,
        }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_directory(
    root: Path,
    expected_files: Iterable[Mapping[str, Any]] = (),
    *,
    verify_lfs_sha256: bool = False,
) -> VerificationResult:
    errors: list[str] = []
    checked: set[Path] = set()

    config_path = root / "config.json"
    if not config_path.is_file():
        errors.append("missing config.json")
    else:
        try:
            json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid config.json: {exc}")
        checked.add(config_path)

    shards = sorted(root.rglob("*.safetensors")) if root.is_dir() else []
    if not shards:
        errors.append("no safetensors model weights found")
    for shard in shards:
        checked.add(shard)
        if shard.stat().st_size <= 0:
            errors.append(f"empty weight shard: {shard.relative_to(root)}")

    for index_path in root.rglob("*.safetensors.index.json") if root.is_dir() else ():
        checked.add(index_path)
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                errors.append(f"invalid weight_map: {index_path.relative_to(root)}")
                continue
            for relative in sorted(set(weight_map.values())):
                shard = index_path.parent / str(relative)
                if not shard.is_file():
                    errors.append(f"index references missing shard: {shard.relative_to(root)}")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid shard index {index_path.relative_to(root)}: {exc}")

    for item in expected_files:
        relative = str(item.get("path", ""))
        if not relative:
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"missing manifest file: {relative}")
            continue
        checked.add(path)
        expected_size = item.get("size")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            errors.append(f"size mismatch: {relative} (expected {expected_size}, got {path.stat().st_size})")
            continue
        expected_sha = item.get("sha256")
        if verify_lfs_sha256 and expected_sha and sha256_file(path).lower() != str(expected_sha).lower():
            errors.append(f"SHA-256 mismatch: {relative}")

    return VerificationResult(
        valid=not errors,
        errors=tuple(errors),
        checked_files=len(checked),
        checked_bytes=sum(path.stat().st_size for path in checked if path.is_file()),
    )


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("download manifest must be a JSON object")
    return value
