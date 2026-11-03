"""录制多个回合并拼成网格视频，用于直观展示策略能力。

单个回合成功只要 9-32 步，逐个看要反复开播放器；拼成网格后可以在同一条时间轴上
对比不同初始条件（这里按目标高度挑选，覆盖从 2 cm 到 32 cm）。

不同回合长度不一致，统一用「成功后的最后一帧定格」补齐到最长回合的长度，
这样所有格子同步推进，不会出现某个格子提前黑屏。

用法（仓库根目录）：
    ./.venv-maniskill/bin/python experiments/04_ppo_pickcube/record_grid.py \
        --checkpoint experiments/04_ppo_pickcube/runs/<run>/final_ckpt.pt \
        --seeds 140,146,114,128,134,116
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401

STAGE_DIR = Path(__file__).resolve().parent
REPO = STAGE_DIR.parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from common.rollout import _scalar, build_agent  # noqa: E402

BASELINE_PY = REPO / "third_party/ManiSkill/examples/baselines/ppo/ppo.py"


def render_frame(env) -> np.ndarray:
    f = env.render()
    if isinstance(f, torch.Tensor):
        f = f.cpu().numpy()
    f = np.asarray(f)
    if f.ndim == 4:
        f = f[0]
    return f.astype(np.uint8)


def tile(frame: np.ndarray, title: str, size: int):
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).resize((size, size), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    # 白色边框：格子之间的桌面色接近，没有边框很难看出分界
    draw.rectangle([0, 0, img.width - 1, img.height - 1],
                   outline=(255, 255, 255), width=2)
    draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    draw.text((4, 2), title, fill=(240, 240, 240))
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser(description="录制多回合并拼成网格视频")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", default="140,146,114,128,134,116",
                    help="逗号分隔的种子列表")
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--control-mode", default="pd_joint_delta_pos")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--size", type=int, default=320, help="每个格子边长（像素）")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        sys.exit(f"checkpoint 不存在: {ckpt}")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) > args.rows * args.cols:
        sys.exit(f"种子数 {len(seeds)} 超过网格容量 {args.rows}x{args.cols}")

    out_dir = Path(args.out) if args.out else ckpt.parent / "grid_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(args.env_id, num_envs=1, obs_mode="state",
                   render_mode="rgb_array", sim_backend="physx_cuda",
                   control_mode=args.control_mode)
    u = env.unwrapped

    obs, _ = env.reset(seed=seeds[0])
    agent = build_agent(ckpt, obs.shape[1:], env.action_space.low.reshape(-1),
                        env.action_space.high.reshape(-1), device, BASELINE_PY)

    episodes: list[dict] = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        goal_z = float(np.asarray(
            u.goal_site.pose.p[0].cpu() if isinstance(u.goal_site.pose.p, torch.Tensor)
            else u.goal_site.pose.p
        ).reshape(-1)[2])

        frames: list[np.ndarray] = []
        success, steps = False, 0
        terminated = truncated = False
        while not (terminated or truncated) and steps < args.max_steps:
            with torch.no_grad():
                action = agent.get_action(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            frames.append(render_frame(env))
            if _scalar(info.get("success", 0.0)) > 0.5:
                success = True
                break

        label = f"seed {seed}  goal_z {goal_z * 100:.1f}cm  {steps}步  {'OK' if success else 'FAIL'}"
        episodes.append({"seed": seed, "frames": frames, "label": label,
                         "success": success, "steps": steps, "goal_z": goal_z})
        print(f"seed={seed} goal_z={goal_z:.3f} steps={steps} success={success}")

    env.close()

    # 统一长度：短的用最后一帧定格，保证所有格子同步
    T = max(len(e["frames"]) for e in episodes)
    for e in episodes:
        while len(e["frames"]) < T:
            e["frames"].append(e["frames"][-1])

    blank = np.zeros((args.size, args.size, 3), dtype=np.uint8)
    grid_frames: list[np.ndarray] = []
    for t in range(T):
        tiles = []
        for i in range(args.rows * args.cols):
            if i < len(episodes):
                e = episodes[i]
                done = min(t + 1, e["steps"])
                title = e["label"] if done >= e["steps"] else \
                    f"seed {e['seed']}  goal_z {e['goal_z'] * 100:.1f}cm  step {done}"
                tiles.append(tile(e["frames"][t], title, args.size))
            else:
                tiles.append(blank)
        rows_img = [np.hstack(tiles[r * args.cols:(r + 1) * args.cols])
                    for r in range(args.rows)]
        grid_frames.append(np.vstack(rows_img))

    # 首尾定格几帧，便于看清初始布局与最终结果
    seq = [grid_frames[0]] * 4 + grid_frames + [grid_frames[-1]] * 8

    import imageio.v2 as imageio

    n_ok = sum(e["success"] for e in episodes)
    path = out_dir / f"grid_{len(episodes)}eps_{n_ok}ok.mp4"
    imageio.mimsave(path, seq, fps=args.fps, macro_block_size=1)
    print(f"\n已保存 {path}")
    print(f"  网格 {args.rows}x{args.cols}，每格 {args.size}px，"
          f"{len(seq)} 帧 @ {args.fps}fps")
    print(f"  成功 {n_ok}/{len(episodes)}，目标高度 "
          f"{min(e['goal_z'] for e in episodes) * 100:.1f}–"
          f"{max(e['goal_z'] for e in episodes) * 100:.1f} cm")


if __name__ == "__main__":
    main()
