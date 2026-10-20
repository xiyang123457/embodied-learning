"""W4 · D2 配图：奇异位的两张图。

用途
    图 1  两个速度箭头怎么从「张开成直角」一路合拢成「几乎平行」
    图 2  逆解 |q̇| 随 θ2 缩小的放大规律，以及它撞上电机转速上限的位置

输入
    无。θ1 = 0，l1 = l2 = 1。

输出
    outputs/figures/W4_D2_1_arrows_collapse.png
    outputs/figures/W4_D2_2_ik_blowup.png

跑完应看到的自检数字
    θ2 = 90°  两列夹角 45.00°   cond = 2.618
    θ2 = 5°   两列夹角  2.49°   cond = 57.3
    θ2 = 1e-4 两列夹角  0.0057° |q̇| = 22360 rad/s
    电机上限 10 rad/s 对应 θ2 ≈ 0.2236
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.jacobian import J_ana, cond_num, det_ana
from src.two_link import fk

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

L1 = L2 = 1.0
C1, C2, DARK = "#c0392b", "#27ae60", "#1f2328"
VMAX = 10.0
# VMAX：float，真实电机转速上限的量级（rad/s）。
#   为什么取 10：不是精确值，是一个「量级」参照线，用来说明数学解和物理可执行之间的距离。
#   典型关节电机在 100–300 rpm 量级，折算约 10–30 rad/s。


def column_angle(theta):
    """返回 J 两列之间的夹角（度）。共线时为 0，这就是奇异。"""
    J = J_ana(theta, L1, L2)
    a, b = J[:, 0], J[:, 1]
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def draw_case(ax, theta2, title):
    """画一个位形：手臂 + 两列的箭头 + 夹角标注。"""
    t1 = 0.0
    j2 = np.array([L1 * np.cos(t1), L1 * np.sin(t1)])
    end = np.array(fk(t1, theta2, L1, L2))
    J = J_ana((t1, theta2), L1, L2)
    v1, v2 = J[:, 0], J[:, 1]

    ax.set_aspect("equal")
    ax.grid(alpha=0.15, linestyle=":")
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(-0.25, 2.35)

    ax.plot([0, j2[0], end[0]], [0, j2[1], end[1]], "-o",
            color=DARK, lw=3.4, ms=8, solid_capstyle="round", zorder=4)
    ax.plot([0, end[0]], [0, end[1]], ":", color=C1, lw=1.5)
    ax.plot([j2[0], end[0]], [j2[1], end[1]], ":", color=C2, lw=1.5)

    ax.annotate("", xy=end + v1, xytext=end,
                arrowprops=dict(arrowstyle="-|>", color=C1, lw=2.8,
                                shrinkA=0, shrinkB=0))
    ax.annotate("", xy=end + v2, xytext=end,
                arrowprops=dict(arrowstyle="-|>", color=C2, lw=2.8,
                                shrinkA=0, shrinkB=0))

    ax.plot(0, 0, "ks", ms=9, zorder=5)
    ax.plot(*end, "o", color=DARK, ms=8, zorder=6)

    ang = column_angle((t1, theta2))
    ax.text(0.02, 2.22,
            f"θ2 = {np.degrees(theta2):.3g}°\n"
            f"两列夹角 = {ang:.3g}°\n"
            f"det(J) = {det_ana(theta2):.3e}\n"
            f"cond(J) = {cond_num((t1, theta2)):.3g}",
            fontsize=9.5, va="top", ha="left", color="#333333",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fbfbf8",
                      edgecolor="#cccccc"))
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("x（米）")
    ax.set_ylabel("y（米）")


def fig_collapse():
    """图 1：两个箭头从张开到合拢。两张必须一起看才有意义。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.4))
    draw_case(axes[0], np.pi / 2, "① θ2 = 90°：两列成 45°，能凑出任意方向")
    draw_case(axes[1], np.deg2rad(5.0), "② θ2 = 5°：两列几乎平行，某个方向推不动了")
    fig.suptitle("奇异不是突然发生的，是两个箭头慢慢合拢的过程\n"
                 "红色第一个关节的列，绿色第二个关节的列，箭头长度就是推得多快",
                 fontsize=12, y=1.02)
    path = OUT / "W4_D2_1_arrows_collapse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_ik_blowup():
    """图 2：逆解 |q̇| 随 θ2 变小而放大，并与电机上限对照。"""
    t2 = np.logspace(-5, np.log10(np.pi / 2), 400)
    # 解析式：q̇ = J⁻¹(1,0)，模长为 sqrt(5)/θ2（θ1=0 时）
    qdot = np.sqrt(5.0) / t2

    fig, ax = plt.subplots(figsize=(9.6, 6.8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, linestyle=":", which="both")
    # 标签里不用 q 上加点的写法：雅黑缺那个组合字形，会渲染成方框
    ax.plot(t2, qdot, color=C1, lw=2.6, label="需要的关节速度 = √5 / θ2")

    # 每个标记单独给标注偏移：最右边那个（θ2=90°）已经贴着坐标轴，
    #   统一按「往右上挪」会把字挤出画布
    marks = [(np.pi / 2, 1.0, "π/2\n1.0", (0.45, 2.2)),
             (0.1, np.sqrt(5) / 0.1, "0.1\n22.3", (1.5, 1.9)),
             (0.01, np.sqrt(5) / 0.01, "0.01\n223", (1.5, 1.9)),
             (1e-4, np.sqrt(5) / 1e-4, "1e-4\n22360", (1.5, 1.9))]
    for x, y, lab, (dx, dy) in marks:
        ax.plot([x], [y], "o", color=C1, ms=8, zorder=5)
        ax.annotate(lab, xy=(x, y), xytext=(x * dx, y * dy), fontsize=9.5,
                    color="#333333")

    ax.axhline(VMAX, color="#d97706", lw=2.0, linestyle="--")
    ax.text(1.2e-5, VMAX * 1.30, "真实电机转速上限（约 10 rad/s）",
            color="#d97706", fontsize=10)

    cross = np.sqrt(5.0) / VMAX
    ax.axvline(cross, color="#d97706", lw=1.2, linestyle=":")
    # 说明文字放在右上方空白处，不画箭头：箭头会横穿整条曲线
    ax.text(cross * 1.25, 3.0e4, f"θ2 < {cross:.4f} 之后\n物理上做不到",
            fontsize=10.5, color="#d97706", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8ec",
                      edgecolor="#d97706"))

    ax.set_xlabel("θ2（弧度，对数轴）")
    ax.set_ylabel("所需关节速度（rad/s，对数轴）")
    ax.set_title("想让末端以 1 m/s 沿手臂方向移动，需要多快的关节\n"
                 "θ2 缩小 10 倍，关节速度涨 10 倍；撞到电机上限之后就是做不到",
                 fontsize=12)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.92)

    path = OUT / "W4_D2_2_ik_blowup.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    # Windows 控制台默认 GBK，打印 √、θ 之外的组合字符会抛 UnicodeEncodeError，
    #   先固定成 utf-8，脚本别在自己打印报告时崩掉
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p1 = fig_collapse()
    p2 = fig_ik_blowup()
    for p in (p1, p2):
        print(f"写出 {p}")

    print("\n== 自检 ==")
    for t2 in [np.pi / 2, np.deg2rad(5.0), 1e-4]:
        print(f"  θ2={t2:.6g}  夹角={column_angle((0.0, t2)):.4g}°  "
              f"det={det_ana(t2):.3e}  cond={cond_num((0.0, t2)):.4g}")
    print(f"  电机上限 10 rad/s 对应 θ2 = {np.sqrt(5.0) / VMAX:.4f}")
    print(f"  θ2=1e-4 时 |qdot| = {np.sqrt(5.0) / 1e-4:.0f} rad/s")


if __name__ == "__main__":
    main()
