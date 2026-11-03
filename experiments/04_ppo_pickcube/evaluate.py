"""阶段 3 第二步：PickCube-v1 独立评估（训练与评估分离）。

PickCube 的 info 里除了 success 还有 is_obj_placed / is_robot_static / is_grasped，
公共评估模块会把这些全部记进 per_episode.csv 与 summary.json，
这样失败时能区分「根本没抓起来」「搬到了但没静止」等不同失败模式。

用法（仓库根目录）：
    ./.venv-maniskill/bin/python experiments/04_ppo_pickcube/evaluate.py \
        --checkpoint experiments/04_ppo_pickcube/runs/<run>/final_ckpt.pt
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
    evaluate_main(STAGE_DIR, BASELINE_PY, default_env_id="PickCube-v1")
