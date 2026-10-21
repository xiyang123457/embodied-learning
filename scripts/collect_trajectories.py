"""W5 · D4：采集位置反馈专家轨迹，一回合一个 npz。

用法（仓库根目录）：
    ./.venv/bin/python scripts/collect_trajectories.py            # k=50, seed 0..19
    ./.venv/bin/python scripts/collect_trajectories.py --k 20 --out data/raw

字段：obs_before / achieved_before / action / obs_after / achieved_after /
      desired_goal / episode_id / step / terminated / truncated / is_success
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from w5_core import SEEDS, collect_episode, make_env, make_policy_feedback

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=float, default=50.0)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()
    policy = make_policy_feedback(args.k)
    try:
        for seed in args.seeds:
            ep = collect_episode(env, policy, seed)
            path = out_dir / f"feedback_seed{seed}.npz"
            np.savez_compressed(path, **ep)
            moved = np.linalg.norm(ep["achieved_after"][-1] - ep["achieved_before"][0])
            final = np.linalg.norm(ep["desired_goal"] - ep["achieved_after"][-1])
            print(f"seed={seed:>2d}  steps={len(ep['step']):>2d}  "
                  f"path_len={moved:.3f} m  final_dist={final:.4f} m  "
                  f"in_tol={ep['is_success'].sum():>2d}  -> {path.name}")
    finally:
        env.close()

    print(f"\n共写入 {len(args.seeds)} 个 npz 到 {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
