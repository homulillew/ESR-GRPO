# 通用脚本

- `prepare_browsecomp_data.py`：把固定研究切分转换成 ECHO/verl parquet。
- `check_echo_compat.py`：只读检查当前 ECHO 快照的上游接口标记。
- `install_echo_hooks.py`：预演或显式安装 ESR-GRPO 的三个 ECHO 钩子。
- `run_all_cpu_checks.ps1`、`run_all_cpu_checks.sh`：运行单元测试和确定性冒烟 episode。

这些脚本服务多个实验，不保存某一条实验线的专属逻辑。
