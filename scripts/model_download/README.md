# 30B 级模型下载

实验计划中的两个大模型是：

- `Qwen/Qwen3-30B-A3B-Instruct-2507`：30B MoE 策略模型，约 60 GB。实验计划中的无后缀名称在代码中映射到这个官方 checkpoint；
- `Qwen/Qwen3-32B`：32B dense Judge，约 64 GB。

下载器先把 revision 解析为不可变 commit，再调用 `huggingface_hub.snapshot_download`。中断后重复执行同一命令即可续传。下载前会估算剩余字节和磁盘空间，完成后检查 `config.json`、全部 safetensors 分片和索引引用，并保存 `download_manifest.json`。

项目实验与训练运行在 Linux 上。模型下载同时维护 Linux Bash 和 Windows PowerShell 入口。

安装：

```bash
pip install -e .
pip install -r requirements-download.txt
```

Linux：

```bash
MODEL_ROOT=/data/models bash scripts/model_download/download_qwen3_30b_a3b_instruct.sh
MODEL_ROOT=/data/models bash scripts/model_download/download_qwen3_32b_judge.sh
```

Windows：

```powershell
.\scripts\model_download\download_qwen3_30b_a3b_instruct.ps1 -DestinationRoot D:\models
.\scripts\model_download\download_qwen3_32b_judge.ps1 -DestinationRoot D:\models
```

私有或限流仓库可在环境变量 `HF_TOKEN` 中提供 token。若网络中断，直接重跑。下载完成后可增加 `--verify-only --verify-lfs-sha256` 做完整 SHA-256 校验；该操作会顺序读取全部权重。

如果模型仓库名称调整，可直接调用通用入口：

```bash
python scripts/model_download/download_hf_model.py \
  --repo-id organization/model --output /data/models/model
```
