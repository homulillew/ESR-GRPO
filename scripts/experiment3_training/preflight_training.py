"""实验三训练前检查：输入数据、模型、ECHO 钩子、GPU 和服务端点。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def endpoint_check(url: str) -> dict[str, Any]:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "esr-grpo-training-preflight/1.0"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return {"ok": 200 <= response.status < 300, "status": response.status}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def check_parquet(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if not path.is_file():
        return result
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
        required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
        result.update(
            rows=len(frame),
            columns=list(frame.columns),
            missing_columns=sorted(required - set(frame.columns)),
            ok=bool(len(frame)) and required.issubset(frame.columns),
        )
    except Exception as exc:
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return result


def check_echo(root: Path) -> dict[str, Any]:
    markers = {
        "verl/trainer/ppo/core_algos.py": ('register_adv_est("esr_grpo")',),
        "verl/trainer/ppo/ray_trainer.py": ("esr_response_credit_mask", "esr_legal_submit"),
        "verl/experimental/agent_loop/tool_agent_loop.py": (
            "esr_token_span",
            "finalize_echo_agent_data",
            "esr_response_credit_mask",
        ),
    }
    missing_files: list[str] = []
    missing_markers: list[str] = []
    for relative, expected in markers.items():
        path = root / relative
        if not path.is_file():
            missing_files.append(relative)
            continue
        source = path.read_text(encoding="utf-8")
        missing_markers.extend(f"{relative}: {marker}" for marker in expected if marker not in source)
    return {
        "root": str(root.resolve()),
        "ok": not missing_files and not missing_markers,
        "missing_files": missing_files,
        "missing_markers": missing_markers,
    }


def check_model(path: Path) -> dict[str, Any]:
    indexes = list(path.glob("*.safetensors.index.json")) if path.is_dir() else []
    shards = list(path.glob("*.safetensors")) if path.is_dir() else []
    result = {
        "path": str(path.resolve()),
        "exists": path.is_dir(),
        "config": (path / "config.json").is_file(),
        "safetensor_files": len(shards),
        "index_files": len(indexes),
    }
    result["ok"] = result["exists"] and result["config"] and bool(shards)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--echo-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--val-file", required=True, type=Path)
    parser.add_argument("--tool-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--required-gpus", type=int, default=4)
    parser.add_argument("--retrieval-health-url")
    parser.add_argument("--verifier-models-url")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    dependencies = {}
    for package in ("torch", "numpy", "ray", "hydra-core", "tensordict", "transformers", "sglang"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
            errors.append(f"missing Python package: {package}")

    gpu: dict[str, Any] = {"available": False, "count": 0, "devices": []}
    try:
        import torch

        gpu["available"] = torch.cuda.is_available()
        gpu["count"] = torch.cuda.device_count()
        gpu["cuda_runtime"] = torch.version.cuda
        gpu["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2),
            }
            for index in range(torch.cuda.device_count())
        ]
    except Exception as exc:
        gpu["error"] = f"{type(exc).__name__}: {exc}"
    if gpu["count"] < args.required_gpus:
        errors.append(f"found {gpu['count']} CUDA GPUs, require {args.required_gpus}")

    echo = check_echo(args.echo_root)
    model = check_model(args.model_path)
    train = check_parquet(args.train_file)
    validation = check_parquet(args.val_file)
    if not echo["ok"]:
        errors.append("ECHO ESR-GRPO hooks are missing")
    if not model["ok"]:
        errors.append("model directory is incomplete")
    if not train.get("ok"):
        errors.append("training parquet is invalid")
    if not validation.get("ok"):
        errors.append("validation parquet is invalid")
    if not args.tool_config.is_file():
        errors.append(f"tool config does not exist: {args.tool_config}")
    if platform.system() != "Linux":
        warnings.append("正式 ECHO/verl 训练应在 Linux CUDA 环境运行")
    if sys.version_info[:2] != (3, 10):
        warnings.append(f"实验锁定 Python 3.10，当前为 {platform.python_version()}")
    java = shutil.which("java")
    if java is None:
        warnings.append("没有找到 Java；启动或重建 ECHO 检索服务时需要 Java 21")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(args.output_dir, os.W_OK):
        errors.append(f"output directory is not writable: {args.output_dir}")
    services = {}
    for name, url in (("retrieval", args.retrieval_health_url), ("verifier", args.verifier_models_url)):
        services[name] = endpoint_check(url) if url else {"ok": None, "skipped": True}
        if url and not services[name]["ok"]:
            errors.append(f"{name} endpoint is unavailable")

    report = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "system": {"platform": platform.platform(), "python": platform.python_version(), "java": java},
        "dependencies": dependencies,
        "gpu": gpu,
        "echo": echo,
        "model": model,
        "data": {"train": train, "validation": validation},
        "tool_config": str(args.tool_config.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "services": services,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
