"""阶段 3 第二步：PickCube-v1 上的 PPO（封装 ManiSkill 官方 baseline）。

具体逻辑在 `experiments/common/runner.py`，本文件只声明本阶段的目录与 baseline 路径。
PickCube 与 PushCube 使用同一套记录规范，两者成功率**不可直接比较**（判据不同）。

用法（仓库根目录）：
    ./.venv-maniskill/bin/python experiments/04_ppo_pickcube/run.py --config smoke
    ./.venv-maniskill/bin/python experiments/04_ppo_pickcube/run.py --config train --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
REPO = STAGE_DIR.parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from common.runner import run_stage  # noqa: E402

BASELINE_PY = REPO / "third_party/ManiSkill/examples/baselines/ppo/ppo.py"

if __name__ == "__main__":
    run_stage(STAGE_DIR, BASELINE_PY)
