"""阶段 3：PushCube-v1 独立评估（训练与评估严格分离）。

为什么不直接用官方 baseline 的 `--evaluate`：
  官方脚本在评估时用 `eval_envs.reset(seed=args.seed)`，即**训练种子**，
  且回合数由 num_eval_envs x num_eval_steps 间接决定。
  计划书第 8 节要求「评估种子列表与训练种子不重叠」「回合数明确」，
  所以这里单独做一次评估。

视频策略（计划书要求"保存成功和失败视频"，不能只留成功的那条）：
  先跑完全部回合拿到逐回合结果，再回头重跑「首个成功」和「首个失败」的种子并录制，
  这样保证两类案例都有视频，也不会为了录视频拖慢整体评估。

本脚本不改第三方代码，只从 third_party 的 ppo.py 动态导入 Agent 网络定义
（复用结构，不执行其 main），然后加载 checkpoint 做确定性推断。

用法（仓库根目录）：
    ./.venv-maniskill/bin/python experiments/03_ppo_pushcube/evaluate.py \
        --checkpoint experiments/03_ppo_pushcube/runs/<run>/final_ckpt.pt
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401  触发任务注册

STAGE_DIR = Path(__file__).resolve().parent
REPO = STAGE_DIR.parents[1]
PPO_PY = REPO / "third_party/ManiSkill/examples/baselines/ppo/ppo.py"


def _scalar(x) -> float:
    """ManiSkill 的 reward/terminated 常是 cuda tensor，必须先搬回 host 再转。"""
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0])
    return float(np.asarray(x).reshape(-1)[0])


class _Shim:
    """Agent 只需要 single_*_space 的 shape，用最小替身避免构造真实环境。"""

    def __init__(self, obs_shape, act_low, act_high):
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, obs_shape, dtype=np.float32)
        self.single_action_space = gym.spaces.Box(
            act_low, act_high, dtype=np.float32)


def build_agent(ckpt: Path, obs_shape, act_low, act_high, device: torch.device):
    """从官方 baseline 动态取 Agent 定义并装载权重。"""
    spec = importlib.util.spec_from_file_location("ms_official_ppo", PPO_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {PPO_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # 只执行模块级 import，不触发 main

    agent = mod.Agent(_Shim(obs_shape, act_low, act_high)).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=device))
    agent.eval()
    return agent


def run_episode(env, agent, seed: int, max_steps: int, frames_out: list | None = None) -> dict:
    """跑一个回合。frames_out 非 None 时顺便收集渲染帧。"""
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    success_ever = False
    terminated = truncated = False

    while not (terminated or truncated) and steps < max_steps:
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += _scalar(reward)
        steps += 1
        if "success" in info:
            success_ever |= _scalar(info["success"]) > 0.5
        if frames_out is not None:
            f = env.render()
            if isinstance(f, torch.Tensor):
                f = f.cpu().numpy()
            f = np.asarray(f)
            if f.ndim == 4:
                f = f[0]
            label = (f"step {steps:02d}/{max_steps}  "
                     f"reward {_scalar(reward):+.3f}  success={success_ever}")
            frames_out.append((f.astype(np.uint8), label))

    return {
        "seed": seed,
        "return": round(total, 4),
        "steps": steps,
        "success": int(success_ever),
        "terminated": int(_scalar(terminated) > 0.5),
        "truncated": int(_scalar(truncated) > 0.5),
    }


def write_video(path: Path, frames: list[tuple[np.ndarray, str]], fps: int,
                hold_first: int = 5, hold_last: int = 10) -> int:
    """把 (帧, 标注) 序列写成慢放视频。

    成功回合只要 11-19 步就结束，按 20 fps 播放不到 1 秒，肉眼根本看不清
    「靠近 → 推动 → 方块进目标圈」的过程；成功时 episode 还会 terminated，
    无法继续录。所以这里做三件事：
      1. 降到低帧率（默认 4 fps）播放
      2. 首帧定格，先让人看清初始布局
      3. 末帧定格，停住看最终结果
    另外每帧左上角叠加步数、当步奖励和成功标志。
    """
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    def annotate(frame: np.ndarray, text: str) -> np.ndarray:
        img = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
        draw.text((4, 2), text, fill=(235, 235, 235))
        return np.asarray(img)

    seq: list[np.ndarray] = []
    seq += [annotate(*frames[0])] * hold_first
    seq += [annotate(*f) for f in frames]
    seq += [annotate(*frames[-1])] * hold_last
    imageio.mimsave(path, seq, fps=fps, macro_block_size=1)
    return len(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description="PushCube PPO 独立评估")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env-id", default="PushCube-v1")
    ap.add_argument("--control-mode", default="pd_joint_delta_pos")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=100,
                    help="评估种子起点，需与训练种子不重叠")
    ap.add_argument("--max-steps", type=int, default=50,
                    help="单回合上限，与训练的 episode horizon 一致")
    ap.add_argument("--no-video", action="store_true", help="完全不录视频")
    ap.add_argument("--video-fps", type=int, default=4,
                    help="评估视频帧率；成功回合只有十几步，默认降到 4fps 慢放")
    ap.add_argument("--out", default=None, help="输出目录，默认放在 checkpoint 同级 eval/")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        sys.exit(f"checkpoint 不存在: {ckpt}")

    out_dir = Path(args.out) if args.out else ckpt.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        render_mode=None if args.no_video else "rgb_array",
        sim_backend="physx_cuda",
        control_mode=args.control_mode,
    )

    obs, _ = env.reset(seed=args.seed_start)
    agent = build_agent(ckpt, obs.shape[1:], env.action_space.low.reshape(-1),
                        env.action_space.high.reshape(-1), device)

    # ---- 1. 全量评估（不录视频，保证速度）
    rows: list[dict] = []
    for i in range(args.n_episodes):
        seed = args.seed_start + i
        row = run_episode(env, agent, seed, args.max_steps)
        rows.append(row)
        print(f"ep {i:02d} seed={seed} return={row['return']:.3f} "
              f"steps={row['steps']} success={bool(row['success'])}")

    # ---- 2. 补录「首个成功」和「首个失败」，两类案例都要有视频
    recorded: list[str] = []
    if not args.no_video:
        cases = []
        ok = next((r for r in rows if r["success"]), None)
        bad = next((r for r in rows if not r["success"]), None)
        if ok:
            cases.append(("success", ok))
        if bad:
            cases.append(("failure", bad))
        if not bad:
            print("[提示] 本次评估没有失败回合，只录到成功视频；不能据此声称策略无失败模式")

        for tag, ref in cases:
            frames: list[tuple[np.ndarray, str]] = []
            redo = run_episode(env, agent, ref["seed"], args.max_steps, frames_out=frames)
            if redo["success"] != ref["success"]:
                print(f"[警告] seed={ref['seed']} 重跑结果与首次不一致 "
                      f"({ref['success']} vs {redo['success']})，检查随机性设置")
            path = out_dir / f"{tag}_seed{ref['seed']}.mp4"
            n_frames = write_video(path, frames, fps=args.video_fps)
            recorded.append(path.name)
            print(f"  录制 {path.name}（{len(frames)} 步 -> {n_frames} 帧 "
                  f"@ {args.video_fps}fps，含首尾定格与标注）")

    env.close()

    for r in rows:
        r["video"] = int(any(r["seed"] == int(n.split("_seed")[1].split(".")[0])
                             for n in recorded))

    with (out_dir / "per_episode.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rets = [r["return"] for r in rows]
    succ = sum(r["success"] for r in rows)
    summary = {
        "checkpoint": str(ckpt),
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "n_episodes": len(rows),
        "seeds": [args.seed_start, args.seed_start + len(rows) - 1],
        "max_steps": args.max_steps,
        "success_count": succ,
        "success_rate": round(succ / len(rows), 4),
        "mean_return": round(statistics.fmean(rets), 4),
        "std_return": round(statistics.stdev(rets), 4) if len(rets) > 1 else 0.0,
        "mean_return_success_only": (
            round(statistics.fmean([r["return"] for r in rows if r["success"]]), 4)
            if succ else None),
        "mean_return_failure_only": (
            round(statistics.fmean([r["return"] for r in rows if not r["success"]]), 4)
            if succ < len(rows) else None),
        "mean_steps": round(statistics.fmean(r["steps"] for r in rows), 2),
        "videos": recorded,
        "device": str(device),
        "note": "deterministic=True, 独立种子列表, 与训练种子不重叠",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"回合数 {summary['n_episodes']}   成功 {succ}/{len(rows)}"
          f"   成功率 {summary['success_rate']:.1%}")
    print(f"平均回报 {summary['mean_return']} ± {summary['std_return']}"
          f"   平均步数 {summary['mean_steps']}")
    if summary["mean_return_failure_only"] is not None:
        print(f"成功回合平均回报 {summary['mean_return_success_only']}   "
              f"失败回合平均回报 {summary['mean_return_failure_only']}")
    print(f"输出 {out_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
