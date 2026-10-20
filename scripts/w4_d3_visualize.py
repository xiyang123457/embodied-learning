"""W4 · D3/D4 配图：PD 的三张图。

用途
    图 1  固定 Kd，改 Kp 三档：变快，但过某个点开始超调
    图 2  固定 Kp，改 Kd 三档：变稳，代价是变慢
    图 3  阻尼比 ζ 随 Kp 的变化，三条线对应三个 Kd —— 这张才看得见耦合

输入
    无。参数取 src/pd_control.py 的默认值（I=0.1, b=0.5, qd=1.0）。

输出
    outputs/figures/W4_D3_1_kp_sweep.png
    outputs/figures/W4_D3_2_kd_sweep.png
    outputs/figures/W4_D3_3_zeta_map.png

跑完应看到的自检数字
    Kp 25/100/400 超调 0.00 / 0.00 / 12.84 %
    Kd 2/6.3/20   超调 25.62 / 0.00 / 0.00 %
    Kd=2 那条线的 ζ 在 Kp≈15.6 处穿过 1
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pd_control import B_DEFAULT, I_DEFAULT, QD_DEFAULT, metrics, simulate, zeta

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = ["#2c6fbb", "#27ae60", "#c0392b"]


def draw_sweep(values, vary, fixed_kp, fixed_kd, title, path_name):
    """画一组阶跃响应曲线。

    vary 取 "kp" 或 "kd"，决定 values 是 Kp 列表还是 Kd 列表。
    """
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    ax.grid(alpha=0.2, linestyle=":")
    ax.axhline(QD_DEFAULT, color="#888888", lw=1.2, linestyle="--")
    ax.axhline(0.9 * QD_DEFAULT, color="#bbbbbb", lw=1.0, linestyle=":")

    vary_label = "Kp" if vary == "kp" else "Kd"
    # vary_label：str，用于图例和表头。不用 vary.upper()，否则会打成 "KP"，与正文的 Kp 不一致
    rows = []
    for v, color in zip(values, COLORS):
        kp = v if vary == "kp" else fixed_kp
        kd = v if vary == "kd" else fixed_kd
        ts, qs = simulate(kp, kd)
        over, rise, ess = metrics(ts, qs)
        z = zeta(kp, kd)
        rows.append((v, z, over, rise, ess))
        ax.plot(ts, qs, color=color, lw=2.4,
                label=f"{vary_label}={v:g}   ζ={z:.2f}")

    ax.text(1.98, QD_DEFAULT + 0.035, "目标 q_d = 1.0", ha="right", fontsize=9.5,
            color="#666666")
    ax.text(1.98, 0.9 * QD_DEFAULT - 0.085, "0.9 q_d（上升时间判据）", ha="right",
            fontsize=9, color="#999999")

    # 表头用纯 ASCII：Consolas 没有中文字形，写「超调」会渲染成方框并刷一堆警告
    head = f"{vary_label:<7} {'zeta':>5}   {'os%':>7}   {'rise s':>6}   {'ess':>9}"
    body = "\n".join(
        f"{v:<7g} {z:5.2f}   {o:7.2f}   {r:6.3f}   {e:9.1e}"
        for v, z, o, r, e in rows
    )
    ax.text(0.02, 0.03, head + "\n" + body, transform=ax.transAxes, fontsize=9.5,
            family="Consolas", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fbfbf8",
                      edgecolor="#cccccc"))

    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.15, 1.45)
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("关节角 q（rad）")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="center right", fontsize=10, framealpha=0.92)

    path = OUT / path_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path, rows


def draw_zeta_map():
    """ζ 随 Kp 的变化：三条线对应三个 Kd，穿过 ζ=1 就进入超调区。"""
    kp = np.linspace(10, 500, 400)
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.grid(alpha=0.2, linestyle=":")
    ax.axhline(1.0, color="#c0392b", lw=1.6, linestyle="--")
    # 说明文字放在虚线右段：左段会和 Kp≈16 那个标记点、以及上升的蓝线撞在一起
    ax.text(210, 1.09, "ζ = 1 临界阻尼，Kp 越过这条线就开始超调", color="#c0392b",
            fontsize=10, va="bottom")

    for kd, color in zip([2.0, 6.3, 20.0], COLORS):
        ax.plot(kp, zeta(kp, kd), color=color, lw=2.4, label=f"Kd = {kd:g}")
        cross = (kd + B_DEFAULT) ** 2 / (4.0 * I_DEFAULT)
        # 解析解：令 (kd+b)/(2 sqrt(Kp I)) = 1，得 Kp = (kd+b)^2 / (4I)
        if 10 < cross < 500:
            ax.plot([cross], [1.0], "o", color=color, ms=9, zorder=5)
            # 标注挪到标记点右侧：cross≈16 时 x*0.62 会跑出左边界，字被切掉
            ax.annotate(f"Kp≈{cross:.0f}", xy=(cross, 1.0),
                        xytext=(cross * 1.35, 1.42), fontsize=9.5, color=color,
                        ha="left")

    ax.set_xlabel("Kp")
    ax.set_ylabel("阻尼比 ζ = (Kd + b) / (2·√(Kp·I))")
    ax.set_title("Kp 和 Kd 不是各调各的：Kp 在分母上\n"
                 "固定 Kd 把 Kp 往上推，ζ 会一路降到 1 以下，开始超调",
                 fontsize=12)
    ax.set_xlim(10, 500)
    ax.set_ylim(0, 4.2)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    path = OUT / "W4_D3_3_zeta_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p1, _ = draw_sweep([25, 100, 400], "kp", 100, 6.3,
                       "固定 Kd = 6.3，改 Kp：从过阻尼一路滑到欠阻尼",
                       "W4_D3_1_kp_sweep.png")
    p2, _ = draw_sweep([2.0, 6.3, 20.0], "kd", 100, 6.3,
                       "固定 Kp = 100，改 Kd：从抖到稳，代价是变慢",
                       "W4_D3_2_kd_sweep.png")
    p3 = draw_zeta_map()

    for p in (p1, p2, p3):
        print(f"写出 {p}")

    print("\n== 自检 ==")
    for kp in [25, 100, 400]:
        ts, qs = simulate(kp, 6.3)
        o, r, e = metrics(ts, qs)
        print(f"  Kp={kp:<4} zeta={zeta(kp, 6.3):.2f}  超调={o:6.2f}%  "
              f"上升={r:.3f}s  稳态={e:.2e}")
    for kd in [2.0, 6.3, 20.0]:
        ts, qs = simulate(100, kd)
        o, r, e = metrics(ts, qs)
        print(f"  Kd={kd:<5g} zeta={zeta(100, kd):.2f}  超调={o:6.2f}%  "
              f"上升={r:.3f}s  稳态={e:.2e}")


if __name__ == "__main__":
    main()
