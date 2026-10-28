"""一键跑完第一~第四步全部实验，并把结果写进 outputs/metrics/。

用法：
    python scripts/run_all.py                 # 完整流程（CPU 上约 10~20 分钟）
    python scripts/run_all.py --quick         # 快速模式：减少轮数，几分钟跑完
    python scripts/run_all.py --skip-step1    # 跳掉最慢的从零训练
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, *cmd], cwd=str(ROOT))
    ok = proc.returncode == 0
    print(f"<<< {label} {'成功' if ok else '失败'}，用时 {time.perf_counter() - t0:.1f}s",
          flush=True)
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="一键运行全部实验")
    p.add_argument("--quick", action="store_true", help="减少训练轮数")
    p.add_argument("--skip-step1", action="store_true")
    p.add_argument("--skip-step3", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    e1 = "10" if args.quick else "30"
    e3 = "5" if args.quick else "10"
    e4 = "3" if args.quick else "5"

    jobs: list[tuple[str, list[str]]] = []
    if not args.skip_step1:
        jobs.append(("第一步 手写AlexNet从零训练",
                     [str(SCRIPTS / "step1_scratch.py"), "--epochs", e1, "--device", args.device]))
    jobs.append(("第二步 加载ImageNet预训练权重",
                 [str(SCRIPTS / "step2_pretrained.py"), "--epochs", e4,
                  "--finetune", "--device", args.device]))
    if not args.skip_step3:
        jobs.append(("第三步 数据增强对照",
                     [str(SCRIPTS / "step3_augmentation.py"), "--epochs", e3, "--device", args.device]))
    jobs.append(("第四步 torchvision模型横向对比",
                 [str(SCRIPTS / "step4_model_zoo.py"), "--epochs", e4, "--device", args.device]))

    started = time.perf_counter()
    failed = [label for label, cmd in jobs if not run(cmd, label)]

    print(f"\n{'=' * 70}")
    print(f"全部完成，总用时 {time.perf_counter() - started:.1f}s")
    if failed:
        print("以下步骤失败：" + "、".join(failed))
        return 1
    print("结果见 outputs/metrics/*.json，权重见 outputs/checkpoints/*.pth")
    print("接下来运行：python backend/server.py  然后浏览器打开 http://127.0.0.1:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
