"""阶段 2/3：ManiSkill 仿真后端（CPU vs GPU）吞吐实测。

回答一个具体问题：ManiSkill 的物理仿真是该走 CPU 还是 GPU？
与 CartPole 的 PPO 不同，这里的瓶颈是物理仿真本身，而 SAPIEN 3 支持
GPU 批量仿真（几百到几千个环境同时步进），所以结论可能相反。

两个必须知道的约束（实测得到，不是文档写的）：
  1. `GPU PhysX can only be enabled once before any other code involving PhysX`
     —— 同一进程内不能先初始化 CPU PhysX 再切 GPU PhysX 后端。
     因此每个配置必须跑在独立子进程里，这正是本脚本用 subprocess 的原因。
  2. SAPIEN/PhysX 的显存不经过 torch 的分配器统计，
     `torch.cuda.max_memory_allocated()` 看不到它，
     必须用 `torch.cuda.mem_get_info()` 读全局剩余显存来估算。

用法（仓库根目录）：
    ./.venv-maniskill/bin/python scripts/bench_maniskill_backend.py
    ./.venv-maniskill/bin/python scripts/bench_maniskill_backend.py --steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")

OUT = Path(__file__).resolve().parent / "outputs"
ENV_ID = "PushCube-v1"
RESULT_PREFIX = "RESULT_JSON:"

CONFIGS: list[tuple[str, int]] = [
    ("cpu", 1),
    ("gpu", 1),
    ("gpu", 64),
    ("gpu", 256),
    ("gpu", 1024),
]


def gpu_used_gb() -> float:
    """读整卡已用显存。PhysX 的分配不经过 torch 分配器，只能看设备级统计。"""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3
    except Exception:
        try:
            import torch

            free, total = torch.cuda.mem_get_info()
            return (total - free) / 1024**3
        except Exception:
            return 0.0


def bench(backend: str, num_envs: int, steps: int) -> dict:
    """在独立进程内跑一个配置，返回吞吐与显存峰值。"""
    import gymnasium as gym
    import torch

    import mani_skill.envs  # noqa: F401  导入才触发任务注册

    base_gb = gpu_used_gb()
    try:
        env = gym.make(ENV_ID, obs_mode="state", num_envs=num_envs, sim_backend=backend)
    except Exception as e:  # noqa: BLE001
        return {"backend": backend, "num_envs": num_envs,
                "error": f"{type(e).__name__}: {e}"}

    try:
        env.reset(seed=0)
        dev = env.unwrapped.device
        # 批量化后 action_space.shape 是 (num_envs, act_dim)，只有最后一维是单环境动作维度
        act_dim = env.action_space.shape[-1]
        act = torch.zeros((num_envs, act_dim), device=dev)

        # 预热：PhysX/内核首次加载开销很大，不能计入吞吐
        for _ in range(10):
            env.step(act)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(act)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        peak_gb = max(gpu_used_gb() - base_gb, 0.0) if dev.type == "cuda" else 0.0

        result = {
            "backend": backend,
            "num_envs": num_envs,
            "device": str(dev),
            "batched_steps": steps,
            "env_steps": steps * num_envs,
            "seconds": round(dt, 3),
            "env_steps_per_s": round(steps * num_envs / dt, 1),
            "peak_gpu_gb": round(peak_gb, 2),
        }
    except Exception as e:  # noqa: BLE001
        result = {"backend": backend, "num_envs": num_envs,
                  "error": f"{type(e).__name__}: {e}"}
    finally:
        env.close()

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="ManiSkill 仿真后端吞吐实测")
    ap.add_argument("--steps", type=int, default=100, help="批量 step 次数")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", type=str, default="cpu", help=argparse.SUPPRESS)
    ap.add_argument("--num-envs", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()

    # 子进程模式：只跑一个配置，把结果写成一行 JSON 交给父进程
    if args.single:
        r = bench(args.backend, args.num_envs, args.steps)
        print(RESULT_PREFIX + json.dumps(r, ensure_ascii=False))
        return

    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"{'backend':<8}{'num_envs':>9}{'env_steps/s':>14}{'峰值显存GB':>12}{'倍数':>10}")
    print("=" * 80)

    rows: list[dict] = []
    baseline: float | None = None
    for backend, n in CONFIGS:
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--single", "--backend", backend,
               "--num-envs", str(n), "--steps", str(args.steps)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        picked = [ln for ln in proc.stdout.splitlines() if ln.startswith(RESULT_PREFIX)]
        if picked:
            r = json.loads(picked[-1][len(RESULT_PREFIX):])
        else:
            err = (proc.stderr or proc.stdout or "unknown").strip().splitlines()
            r = {"backend": backend, "num_envs": n,
                 "error": err[-1] if err else "no output"}

        rows.append(r)
        if "error" in r:
            print(f"{backend:<8}{n:>9}{'—':>14}{'—':>12}{'失败':>10}")
            print(f"         {r['error'][:110]}")
        else:
            if baseline is None:
                baseline = r["env_steps_per_s"]
            speedup = r["env_steps_per_s"] / baseline
            print(f"{backend:<8}{n:>9}{r['env_steps_per_s']:>14.1f}"
                  f"{r['peak_gpu_gb']:>12.2f}{speedup:>9.1f}x")

    print("=" * 80)

    path = OUT / "backend_bench.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {path}")


if __name__ == "__main__":
    main()
