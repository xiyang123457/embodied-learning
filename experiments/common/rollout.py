"""阶段实验的通用独立评估（训练与评估分离）。

为什么不用官方 baseline 的 --evaluate：
  官方脚本用 `eval_envs.reset(seed=args.seed)`，即**训练种子**，
  且回合数由 num_eval_envs x num_eval_steps 间接决定。
  计划书第 8 节要求「评估种子列表与训练种子不重叠」「回合数明确」。

设计要点：
  1. 不修改第三方代码，只从 baseline 的 ppo.py 动态导入 Agent 网络定义
  2. 自动收集 info 里所有布尔指标（PickCube 有 success / is_obj_placed /
     is_robot_static / is_grasped），同时记 ever 与 end 两个口径
  3. 视频先跑完再补录「首个成功」和「首个失败」，两类都留；
     成功回合只有十几步且成功即终止，所以默认低帧率慢放 + 首尾定格
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


def _scalar(x) -> float:
    """ManiSkill 的 reward/terminated 常是 cuda tensor，必须先搬回 host 再转。"""
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0])
    return float(np.asarray(x).reshape(-1)[0])


def bool_metrics(info: dict) -> dict[str, float]:
    """把 info 里的布尔量抽成 {name: 0/1}，用于逐回合统计。"""
    out: dict[str, float] = {}
    for key, value in info.items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.bool:
            out[key] = float(value.reshape(-1)[0].item())
        elif isinstance(value, (bool, np.bool_)):
            out[key] = float(value)
    return out


class _Shim:
    """Agent 只需要 single_*_space 的 shape，用最小替身避免构造真实环境。"""

    def __init__(self, obs_shape, act_low, act_high):
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, obs_shape, dtype=np.float32)
        self.single_action_space = gym.spaces.Box(
            act_low, act_high, dtype=np.float32)


def build_agent(ckpt: Path, obs_shape, act_low, act_high,
                device: torch.device, baseline_py: Path):
    """从官方 baseline 动态取 Agent 定义并装载权重。"""
    spec = importlib.util.spec_from_file_location("ms_official_ppo", baseline_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {baseline_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # 只执行模块级 import，不触发 main

    agent = mod.Agent(_Shim(obs_shape, act_low, act_high)).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=device))
    agent.eval()
    return agent


def run_episode(env, agent, seed: int, max_steps: int,
                frames_out: list | None = None) -> dict:
    """跑一个回合，返回逐回合记录；frames_out 非 None 时收集 (帧, 标注)。"""
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    ever: dict[str, float] = {}
    end: dict[str, float] = {}
    terminated = truncated = False

    while not (terminated or truncated) and steps < max_steps:
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += _scalar(reward)
        steps += 1

        m = bool_metrics(info)
        end = m
        for k, v in m.items():
            ever[k] = max(ever.get(k, 0.0), v)

        if frames_out is not None:
            f = env.render()
            if isinstance(f, torch.Tensor):
                f = f.cpu().numpy()
            f = np.asarray(f)
            if f.ndim == 4:
                f = f[0]
            flags = " ".join(f"{k}={int(v)}" for k, v in sorted(m.items()))
            label = f"step {steps:02d}/{max_steps}  reward {_scalar(reward):+.3f}  {flags}"
            frames_out.append((f.astype(np.uint8), label))

    row: dict = {
        "seed": seed,
        "return": round(total, 4),
        "steps": steps,
        "terminated": int(_scalar(terminated) > 0.5),
        "truncated": int(_scalar(truncated) > 0.5),
    }
    for k in sorted(ever):
        row[f"{k}_ever"] = int(ever[k] > 0.5)
        row[f"{k}_end"] = int(end.get(k, 0.0) > 0.5)
    return row


def write_video(path: Path, frames: list[tuple[np.ndarray, str]], fps: int,
                hold_first: int = 5, hold_last: int = 10) -> int:
    """把 (帧, 标注) 序列写成慢放视频，首尾定格。

    成功回合只要十几步就结束且成功即 terminated，按 20 fps 播放不到 1 秒，
    肉眼看不清过程，所以默认降到低帧率并定格首尾。
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


def evaluate_main(stage_dir: Path, baseline_py: Path,
                  argv: list[str] | None = None,
                  default_env_id: str = "PushCube-v1") -> None:
    stage_dir = stage_dir.resolve()

    ap = argparse.ArgumentParser(description=f"{stage_dir.name} 独立评估")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env-id", default=default_env_id)
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
    args = ap.parse_args(argv)

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
                        env.action_space.high.reshape(-1), device, baseline_py)

    # ---- 1. 全量评估（不录视频，保证速度）
    rows: list[dict] = []
    for i in range(args.n_episodes):
        seed = args.seed_start + i
        row = run_episode(env, agent, seed, args.max_steps)
        rows.append(row)
        keys = " ".join(f"{k}={v}" for k, v in row.items()
                        if k.endswith("_ever"))
        print(f"ep {i:02d} seed={seed} return={row['return']:.3f} "
              f"steps={row['steps']} {keys}")

    # ---- 2. 补录首个成功与首个失败，两类都要有视频
    recorded: list[str] = []
    if not args.no_video:
        cases = []
        ok = next((r for r in rows if r.get("success_ever")), None)
        bad = next((r for r in rows if not r.get("success_ever")), None)
        if ok:
            cases.append(("success", ok))
        if bad:
            cases.append(("failure", bad))
        if not ok:
            print("[提示] 本次评估没有任何成功回合；只录到失败视频，不能声称复现成功")
        if not bad:
            print("[提示] 本次评估没有失败回合，只录到成功视频；不能声称策略无失败模式")

        for tag, ref in cases:
            frames: list[tuple[np.ndarray, str]] = []
            redo = run_episode(env, agent, ref["seed"], args.max_steps,
                               frames_out=frames)
            if redo["success_ever"] != ref["success_ever"]:
                print(f"[警告] seed={ref['seed']} 重跑结果与首次不一致，检查随机性设置")
            path = out_dir / f"{tag}_seed{ref['seed']}.mp4"
            n_frames = write_video(path, frames, fps=args.video_fps)
            recorded.append(path.name)
            print(f"  录制 {path.name}（{len(frames)} 步 -> {n_frames} 帧 "
                  f"@{args.video_fps}fps）")

    env.close()

    for r in rows:
        r["video"] = int(any(r["seed"] == int(n.split("_seed")[1].split(".")[0])
                             for n in recorded))

    with (out_dir / "per_episode.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    returns = [r["return"] for r in rows]
    summary: dict = {
        "checkpoint": str(ckpt),
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "n_episodes": len(rows),
        "seeds": [args.seed_start, args.seed_start + len(rows) - 1],
        "max_steps": args.max_steps,
        "mean_return": round(statistics.fmean(returns), 4),
        "std_return": round(statistics.stdev(returns), 4) if len(returns) > 1 else 0.0,
        "mean_steps": round(statistics.fmean(r["steps"] for r in rows), 2),
        "videos": recorded,
        "device": str(device),
        "note": "deterministic=True, 独立种子列表, 与训练种子不重叠",
    }

    # 每个 *_ever 指标都给出成功数与均值回报分组，便于定位失败模式
    for key in [k for k in rows[0] if k.endswith("_ever")]:
        n_ok = sum(r[key] for r in rows)
        summary[f"{key}_count"] = int(n_ok)
        summary[f"{key}_rate"] = round(n_ok / len(rows), 4)
        if 0 < n_ok < len(rows):
            summary[f"mean_return_when_{key}"] = round(
                statistics.fmean([r["return"] for r in rows if r[key]]), 4)
            summary[f"mean_return_when_not_{key}"] = round(
                statistics.fmean([r["return"] for r in rows if not r[key]]), 4)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"回合数 {summary['n_episodes']}   平均回报 {summary['mean_return']}"
          f" ± {summary['std_return']}   平均步数 {summary['mean_steps']}")
    for key in [k for k in rows[0] if k.endswith("_ever")]:
        print(f"  {key:<20} {summary[f'{key}_count']}/{summary['n_episodes']}"
              f"  ({summary[f'{key}_rate']:.1%})")
    print(f"输出 {out_dir}")
    print("=" * 70)
