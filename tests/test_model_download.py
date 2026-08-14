from __future__ import annotations

import json

from esr_grpo.model_download import sha256_file, verify_model_directory


def test_verify_model_directory_checks_shards_and_manifest(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"safetensor-placeholder")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": shard.name}}), encoding="utf-8"
    )
    expected = [
        {"path": "config.json", "size": 2, "sha256": sha256_file(tmp_path / "config.json")},
        {"path": shard.name, "size": shard.stat().st_size, "sha256": sha256_file(shard)},
    ]
    report = verify_model_directory(tmp_path, expected, verify_lfs_sha256=True)
    assert report.valid is True
    assert report.checked_files == 3


def test_verify_model_directory_reports_missing_index_shard(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "present.safetensors").write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "missing.safetensors"}}), encoding="utf-8"
    )
    report = verify_model_directory(tmp_path)
    assert report.valid is False
    assert any("missing shard" in error for error in report.errors)
