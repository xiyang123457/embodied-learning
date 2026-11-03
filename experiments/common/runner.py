"""阶段实验的通用运行封装：配置驱动 + 每次 run 留痕。

不修改第三方 baseline，只做三件工程上的事：
  1. 实验参数固化在 <stage>/configs/*.json，避免命令行参数漂移
  2. 每次 run 留一份可追溯记录：config 快照、完整命令、包版本、
     第三方仓库 commit、起止时间、耗时、退出码
  3. 输出统一落在 <stage>/runs/<tag>_<timestamp>_seed<N>/

约定（所有阶段一致）：
  - 配置键里 tag / description 是本项目自用，不传给 baseline
  - 配置键的 `_` 在传给 tyro 时换成 `-`；布尔值用 --flag / --no-flag
  - baseline 脚本的 runs/ 相对 cwd 生成，所以必须把 cwd 固定在阶段目录
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# 只在本项目内部使用、不传给 baseline 的配置键
LOCAL_ONLY_KEYS = {"tag", "description", "notes"}


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


def build_cmd(baseline_py: Path, cfg: dict, run_name: str) -> list[str]:
    cmd = [sys.executable, str(baseline_py)]
    for key, value in cfg.items():
        if key in LOCAL_ONLY_KEYS:
            continue
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            cmd.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            cmd += [f"--{flag}", str(value)]
    cmd += ["--exp-name", run_name]
    return cmd


def run_stage(stage_dir: Path, baseline_py: Path, argv: list[str] | None = None) -> None:
    """跑一次阶段实验。

    Args:
        stage_dir: 阶段目录（如 experiments/03_ppo_pushcube），configs/ 与 runs/ 都在它下面
        baseline_py: 第三方 baseline 入口脚本（如 third_party/.../ppo.py）
        argv: 命令行参数，默认 sys.argv[1:]
    """
    stage_dir = stage_dir.resolve()
    repo = stage_dir.parents[1]

    ap = argparse.ArgumentParser(description=f"{stage_dir.name} 运行封装")
    ap.add_argument("--config", required=True, help="configs/ 下的配置名（不含 .json）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令与配置，不执行")
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V",
                    help="临时覆盖配置项，例如 --set num_envs=256")
    args = ap.parse_args(argv)

    config_dir = stage_dir / "configs"
    cfg_path = config_dir / f"{args.config}.json"
    if not cfg_path.exists():
        sys.exit(f"找不到配置: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    for item in args.set:
        if "=" not in item:
            sys.exit(f"--set 需要 K=V 形式，收到: {item}")
        k, v = item.split("=", 1)
        try:
            cfg[k] = json.loads(v)  # 保持 true/false/数字的类型
        except json.JSONDecodeError:
            cfg[k] = v

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.get('tag', 'run')}_{stamp}_seed{cfg.get('seed', 0)}"
    run_dir = stage_dir / "runs" / run_name
    cmd = build_cmd(baseline_py, cfg, run_name)

    if args.dry_run:
        print("run_name:", run_name)
        print("cwd     :", stage_dir)
        print("cmd     :", " \\\n  ".join(cmd))
        return

    if not baseline_py.exists():
        sys.exit(f"找不到官方 baseline: {baseline_py}\n"
                 f"请先执行 sparse clone，见 {stage_dir.name}/README.md")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "run_name": run_name,
        "stage": stage_dir.name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": cmd,
        "cwd": str(stage_dir),
        "config_file": str(cfg_path.relative_to(repo)),
        "config": cfg,
        "versions": env_versions(),
        "git": {
            "main_repo": git_commit(repo),
            "maniskill_baseline": git_commit(repo / "third_party/ManiSkill"),
            "baseline_script": str(baseline_py.relative_to(repo)),
        },
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 74)
    print(f"run     : {run_name}")
    print(f"config  : {cfg_path.name}")
    print(f"cwd     : {stage_dir}")
    print(f"cmd     : {' '.join(cmd)}")
    print("=" * 74)

    log_path = run_dir / "stdout.log"
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd, cwd=str(stage_dir), stdout=subprocess.PIPE,
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
