"""阶段 3：PushCube-v1 上的 PPO（封装 ManiSkill 官方 baseline）。

不修改第三方代码，只做三件工程上的事：
  1. 实验参数固化在 configs/*.json，避免命令行参数漂移、也便于复现
  2. 每次 run 留一份可追溯记录：config 快照、完整命令、包版本、
     第三方仓库 commit、起止时间、耗时、退出码
  3. 输出统一落在 runs/<tag>_<timestamp>_seed<N>/

官方 baseline 位于 third_party/ManiSkill/examples/baselines/ppo/ppo.py
（sparse checkout，commit 见每次 run 的 meta.json）。
注意该脚本的 `runs/` 是**相对 cwd** 生成的，且 sim_backend 硬编码为
`physx_cuda`，所以必须以本阶段目录为 cwd 调用。

用法（仓库根目录）：
    ./.venv-maniskill/bin/python experiments/03_ppo_pushcube/run.py --config smoke
    ./.venv-maniskill/bin/python experiments/03_ppo_pushcube/run.py --config train --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
REPO = STAGE_DIR.parents[1]
PPO_PY = REPO / "third_party/ManiSkill/examples/baselines/ppo/ppo.py"
CONFIG_DIR = STAGE_DIR / "configs"
RUNS_DIR = STAGE_DIR / "runs"

# 这些键只用于本脚本，不传给官方 baseline
LOCAL_ONLY_KEYS = {"tag", "description"}


def git_commit(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def env_versions() -> dict:
    info: dict = {"python": sys.version.split()[0]}
    for pkg in ("torch", "mani_skill", "sapien", "gymnasium", "numpy"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:  # noqa: BLE001
            info[pkg] = "n/a"
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:  # noqa: BLE001
        pass
    return info


def build_cmd(cfg: dict, run_name: str) -> list[str]:
    # tyro 生成的 CLI 是 kebab-case；bool 走 --flag / --no-flag，不接受显式 True/False
    cmd = [sys.executable, str(PPO_PY)]
    for key, value in cfg.items():
        if key in LOCAL_ONLY_KEYS:
            continue
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            cmd.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            cmd += [f"--{flag}", str(value)]
    # exp-name 决定官方脚本的 run_name，进而决定 runs/ 下的目录名
    cmd += ["--exp-name", run_name]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段 3：PushCube PPO 运行封装")
    ap.add_argument("--config", required=True, help="configs/ 下的配置名（不含 .json）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令与配置，不执行")
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V",
                    help="临时覆盖配置项，例如 --set num_envs=256")
    args = ap.parse_args()

    cfg_path = CONFIG_DIR / f"{args.config}.json"
    if not cfg_path.exists():
        sys.exit(f"找不到配置: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    for item in args.set:
        if "=" not in item:
            sys.exit(f"--set 需要 K=V 形式，收到: {item}")
        k, v = item.split("=", 1)
        try:
            cfg[k] = json.loads(v)  # 让 true/false/数字保持类型
        except json.JSONDecodeError:
            cfg[k] = v

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.get('tag', 'run')}_{stamp}_seed{cfg.get('seed', 0)}"
    run_dir = RUNS_DIR / run_name
    cmd = build_cmd(cfg, run_name)

    if args.dry_run:
        print("run_name:", run_name)
        print("cwd     :", STAGE_DIR)
        print("cmd     :", " \\\n  ".join(cmd))
        return

    if not PPO_PY.exists():
        sys.exit(f"找不到官方 baseline: {PPO_PY}\n"
                 "请先执行 sparse clone（见 experiments/03_ppo_pushcube/README.md）")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "run_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": cmd,
        "cwd": str(STAGE_DIR),
        "config_file": str(cfg_path.relative_to(REPO)),
        "config": cfg,
        "versions": env_versions(),
        "git": {
            "main_repo": git_commit(REPO),
            "maniskill_baseline": git_commit(REPO / "third_party/ManiSkill"),
            "baseline_script": str(PPO_PY.relative_to(REPO)),
        },
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 74)
    print(f"run     : {run_name}")
    print(f"config  : {cfg_path.name}")
    print(f"cwd     : {STAGE_DIR}")
    print(f"cmd     : {' '.join(cmd)}")
    print("=" * 74)

    log_path = run_dir / "stdout.log"
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd, cwd=str(STAGE_DIR), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        code = proc.wait()
    elapsed = time.perf_counter() - t0

    meta.update({
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": code,
    })
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总本次产物，便于事后核对有没有 checkpoint / 视频
    artifacts = sorted(
        str(p.relative_to(run_dir))
        for p in run_dir.rglob("*")
        if p.is_file() and p.name not in ("meta.json", "config_snapshot.json", "stdout.log")
    )
    (run_dir / "artifacts.txt").write_text("\n".join(artifacts) + "\n", encoding="utf-8")

    print("=" * 74)
    print(f"耗时 {elapsed:.1f} 秒   退出码 {code}")
    print(f"产物 {len(artifacts)} 个 -> {run_dir}")
    for a in artifacts:
        print(f"  {a}")
    print("=" * 74)

    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
