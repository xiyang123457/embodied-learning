"""W6 · D1/D2：sanity check + 正式训练（3 个种子，按验证集选 checkpoint）。

用法（仓库根目录）：
    ./.venv/bin/python scripts/train_bc.py --sanity                 # 32 样本过拟合检查
    ./.venv/bin/python scripts/train_bc.py --data-size 20           # 20 条轨迹，3 个种子
    ./.venv/bin/python scripts/train_bc.py --data-size 40           # 40 条轨迹（追加 20 条）

产出：
    outputs/w6/train_curves_{n}.csv        每 10 epoch 的 train/val loss（含 seed 列）
    outputs/w6/ckpt/bc_n{n}_seed{s}.pt      模型 + 归一化统计量
    outputs/w6/train_summary.csv           每个种子：最佳 epoch、最佳 val、末 epoch 的 train/val
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bc_dataset import (  # noqa: E402
    EXTRA_SEEDS,
    TRAIN_SEEDS,
    VAL_SEEDS,
    Normalizer,
    episodes_to_arrays,
    load_episodes,
    sample_weights,
)
from src.bc_model import BCPolicy  # noqa: E402

OUT = ROOT / "outputs" / "w6"
CKPT = OUT / "ckpt"


def build_data(data_size: int, weight_static: bool):
    train_seeds = list(TRAIN_SEEDS) + (list(EXTRA_SEEDS) if data_size >= 40 else [])
    train = episodes_to_arrays(load_episodes(train_seeds))
    val = episodes_to_arrays(load_episodes(VAL_SEEDS))

    norm = Normalizer.fit(train["x"])  # 只用训练集
    w = sample_weights(train["dist"]) if weight_static else np.ones(len(train["dist"]), np.float32)
    print(f"数据：训练 {len(train_seeds)} 条 / {len(train['x'])} 样本，"
          f"验证 {len(VAL_SEEDS)} 条 / {len(val['x'])} 样本")
    print(f"      静止样本（dist<0.05）{(train['dist'] < 0.05).mean():.1%}，"
          f"距离≥5cm 的样本 {int((train['dist'] > 0.05).sum())} 个，"
          f"加权={'开(1.0/0.1)' if weight_static else '关'}")
    return train, val, norm, w


def train_one(seed: int, train, val, norm, w, epochs: int, lr: float, batch: int,
              log_every: int = 10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    xt = torch.as_tensor(norm.transform(train["x"]))
    yt = torch.as_tensor(train["y"])
    wt = torch.as_tensor(w)
    xv = torch.as_tensor(norm.transform(val["x"]))
    yv = torch.as_tensor(val["y"])
    # 验证 loss 与训练 loss 用同一加权定义：否则 95% 的近零样本会把远处的误差淹掉
    wv = torch.as_tensor(sample_weights(val["dist"]))
    far_mask = torch.as_tensor(val["dist"] > 0.05)

    model = BCPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(xt)
    hist: list[dict[str, Any]] = []
    best: dict[str, Any] = {"val": float("inf"), "epoch": -1, "state": None,
                            "train": float("nan")}

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total, count = 0.0, 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = model(xt[idx])
            wb = wt[idx].unsqueeze(1)
            loss = ((pred - yt[idx]) ** 2 * wb).sum() / wb.sum()  # 加权 MSE
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        train_loss = total / count

        model.eval()
        with torch.no_grad():
            pv = model(xv)
            sq = (pv - yv) ** 2
            val_loss = float((sq * wv.unsqueeze(1)).sum() / wv.sum())
            mse_all = float(sq.mean())
            mse_far = float(sq[far_mask].mean()) if bool(far_mask.any()) else float("nan")

        if val_loss < best["val"]:
            best = {"val": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict()),
                    "train": train_loss}

        if epoch % log_every == 0 or epoch == 1:
            hist.append({"seed": seed, "epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_mse_all": mse_all, "val_mse_far": mse_far})

    model.load_state_dict(best["state"])
    return model, hist, best


def save_checkpoint(model: BCPolicy, norm: Normalizer, path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **norm.to_dict(), "meta": meta}, path)


# ------------------------------------------------------------------ sanity
def sanity_check(data_size: int = 20, n_samples: int = 32, steps: int = 2000, lr: float = 1e-3) -> None:
    train, _val, norm, _w = build_data(data_size, weight_static=False)
    x = torch.as_tensor(norm.transform(train["x"][:n_samples]))
    y = torch.as_tensor(train["y"][:n_samples])

    torch.manual_seed(0)
    model = BCPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"\n[sanity] {n_samples} 个样本，全批量，Adam lr={lr}，{steps} 步，目标 loss < 1e-4")
    for step in range(1, steps + 1):
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (1, 10, 100, 500, 1000, 2000) or step == steps:
            print(f"  step {step:>5d}  loss {loss.item():.3e}")
    final = nn.functional.mse_loss(model(x), y).item()
    status = "PASS" if final < 1e-4 else ("接近" if final < 1e-2 else "FAIL")
    print(f"[sanity] 最终 loss = {final:.3e}  → {status}")
    print(f"[sanity] 记录用：{n_samples} 样本 / {steps} 步 / final loss {final:.2e} / 参数量 {model.n_params}")


# ------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-size", type=int, default=20, choices=[20, 40])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--no-weight", action="store_true", help="关掉静止样本加权（做对照用）")
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    if args.sanity:
        sanity_check(data_size=args.data_size, lr=args.lr)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    train, val, norm, w = build_data(args.data_size, weight_static=not args.no_weight)

    tag = f"n{args.data_size}" + ("_noweight" if args.no_weight else "")
    all_hist, summary = [], []
    for seed in args.seeds:
        model, hist, best = train_one(seed, train, val, norm, w, args.epochs, args.lr, args.batch)
        all_hist.extend(hist)
        last = hist[-1]
        path = CKPT / f"bc_{tag}_seed{seed}.pt"
        save_checkpoint(model, norm, path,
                        {"seed": seed, "data_size": args.data_size, "epochs": args.epochs,
                         "best_epoch": best["epoch"], "best_val": best["val"]})
        summary.append({
            "tag": tag, "seed": seed, "best_epoch": best["epoch"],
            "best_val": best["val"], "train_at_best": best["train"],
            "train_last": last["train_loss"], "val_last": last["val_loss"],
            "n_params": model.n_params,
        })
        print(f"  seed={seed}: 最佳 epoch {best['epoch']:>3d}  val {best['val']:.3e}  "
              f"(该处 train {best['train']:.3e})；末 epoch train {last['train_loss']:.3e} "
              f"val {last['val_loss']:.3e}")
        print(f"            → {path.relative_to(ROOT)}")

    hist_path = OUT / f"train_curves_{tag}.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tag", "seed", "epoch", "train_loss", "val_loss",
                           "val_mse_all", "val_mse_far"])
        writer.writeheader()
        for r in all_hist:
            writer.writerow({"tag": tag, **r})

    sum_path = OUT / "train_summary.csv"
    exists = sum_path.exists()
    with sum_path.open("a" if exists else "w", newline="", encoding="utf-8") as f:
        fields = ["tag", "seed", "best_epoch", "best_val", "train_at_best", "train_last", "val_last", "n_params"]
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(summary)

    print(f"\n写入 {hist_path.relative_to(ROOT)} 与 {sum_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
