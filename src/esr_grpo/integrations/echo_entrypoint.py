"""注册 ESR 优势函数后进入 ECHO 的 Hydra 训练入口。"""

from __future__ import annotations

from .echo import register_with_echo


def main() -> None:
    register_with_echo()
    from verl.trainer.main_ppo import main as echo_main

    echo_main()


if __name__ == "__main__":
    main()
