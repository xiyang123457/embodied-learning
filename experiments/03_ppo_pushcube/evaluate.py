"""阶段 3：PushCube-v1 独立评估（训练与评估分离）。

具体逻辑在 `experiments/common/rollout.py`，本文件只声明本阶段的目录、baseline
路径与任务 ID。评估用与训练不重叠的种子列表（默认 100-119），并同时保存
「首个成功」和「首个失败」两段慢放视频。

用法（仓库根目录）：
    ./.venv-maniskill/bin/python experiments/03_ppo_pushcube/evaluate.py \
        --checkpoint experiments/03_ppo_pushcube/runs/<run>/final_ckpt.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
REPO = STAGE_DIR.parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from common.rollout import evaluate_main  # noqa: E402

BASELINE_PY = REPO / "third_party/ManiSkill/examples/baselines/ppo/ppo.py"

if __name__ == "__main__":
    evaluate_main(STAGE_DIR, BASELINE_PY, default_env_id="PushCube-v1")
