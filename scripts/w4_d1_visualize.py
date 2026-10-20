"""W4 · D1 配图：雅可比的两列，就是两条力臂各自旋转 90°。

用途
    左边画几何，右边写计算，两边逐行对应。想让人看懂的是这条推导：
        力臂 = 转轴到末端的向量
        第 j 列 = 力臂旋转 90°（因为绕轴转动的速度必然垂直于半径）
    而不是只给一个矩阵让人背。

输入
    无。固定取 theta = (0.5, 0.9)，l1 = l2 = 1，一个一般位形。

输出
    outputs/figures/W4_D1_1_jacobian_columns.png

跑完应看到的自检数字
    末端坐标          (1.0475, 1.4649)
    力臂 2（肘→末端） (0.1700, 0.9854)
    第 1 列 = 旋转90°(1.0475, 1.4649) = (-1.4649, 1.0475)
    第 2 列 = 旋转90°(0.1700, 0.9854) = (-0.9854, 0.1700)
    |第 1 列| = 1.8009   |第 2 列| = 1.0000
    两列与各自力臂点乘 = 0
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.jacobian import J_ana
from src.two_link import fk

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

T1, T2 = 0.5, 0.9
L1 = L2 = 1.0

C_RED, C_GREEN, C_DARK = "#c0392b", "#27ae60", "#1f2328"


def rot90(v):
    """逆时针旋转 90°：(x, y) -> (-y, x)。雅可比每一列就是这个操作。"""
    return np.array([-v[1], v[0]])


def draw_matrix(ax, xc, yc, data, cell=1.15):
    """在 (xc, yc) 居中画一个带方括号的数字矩阵。"""
    rows, cols = data.shape
    w, h = cols * cell, rows * cell
    x0, y0 = xc - w / 2, yc - h / 2

    for r in range(rows):
        for c in range(cols):
            # 只保留 3 位小数：4 位时 "-1.4649" 这个字符串比格宽，相邻两格会粘在一起
            ax.text(x0 + (c + 0.5) * cell, y0 + (rows - 1 - r + 0.5) * cell,
                    f"{data[r, c]:+.3f}", ha="center", va="center",
                    fontsize=10, family="Consolas")

    pad = 0.16
    for xb, sgn in ((x0 - pad, 1), (x0 + w + pad, -1)):
        ax.plot([xb, xb], [y0 - pad, y0 + h + pad], color=C_DARK, lw=1.5)
        ax.plot([xb, xb + 0.16 * sgn], [y0 + h + pad, y0 + h + pad],
                color=C_DARK, lw=1.5)
        ax.plot([xb, xb + 0.16 * sgn], [y0 - pad, y0 - pad],
                color=C_DARK, lw=1.5)
    return y0


def main():
    joint2 = np.array([L1 * np.cos(T1), L1 * np.sin(T1)])
    end = np.array(fk(T1, T2, L1, L2))
    J = J_ana((T1, T2), L1, L2)
    v1, v2 = J[:, 0], J[:, 1]
    arm2 = end - joint2

    fig = plt.figure(figsize=(14.6, 8.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.10, 1.0], wspace=0.02)
    ax = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1])

    # ==== 左：几何 ====
    ax.set_aspect("equal")
    ax.grid(alpha=0.14, linestyle=":")
    ax.set_xlim(-1.85, 1.75)
    ax.set_ylim(-0.35, 2.85)

    # 杆 1 的延长线：θ2 是从这条线量起的，不画出来就不知道 0.9 指的是哪一段
    ext = joint2 + 0.42 * np.array([np.cos(T1), np.sin(T1)])
    ax.plot([joint2[0], ext[0]], [joint2[1], ext[1]], ":", color="#9aa0a6", lw=1.4)

    ax.plot([0, joint2[0], end[0]], [0, joint2[1], end[1]], "-o",
            color=C_DARK, lw=3.6, ms=9, solid_capstyle="round", zorder=4)
    ax.plot([0, end[0]], [0, end[1]], ":", color=C_RED, lw=1.8, zorder=2)
    ax.plot([joint2[0], end[0]], [joint2[1], end[1]], ":", color=C_GREEN, lw=1.8,
            zorder=2)

    ax.add_patch(Arc((0, 0), 0.92, 0.92, theta1=0, theta2=np.degrees(T1),
                     color=C_RED, lw=2.0))
    # θ1 的标注放在弧线外侧、手臂下方：直接放弧中间会被杆 1 穿过
    ax.text(0.66, 0.06, "θ1 = 0.5", color=C_RED, fontsize=10.5,
            fontweight="bold", ha="left", va="center")
    ax.add_patch(Arc(tuple(joint2), 0.86, 0.86,
                     theta1=np.degrees(T1), theta2=np.degrees(T1 + T2),
                     color=C_GREEN, lw=2.0))
    ax.text(joint2[0] + 0.26, joint2[1] + 0.40, "θ2 = 0.9", color=C_GREEN,
            fontsize=10.5, fontweight="bold", ha="left")

    tip1, tip2 = end + v1, end + v2
    ax.annotate("", xy=tip1, xytext=end,
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=3.0,
                                shrinkA=0, shrinkB=0))
    ax.annotate("", xy=tip2, xytext=end,
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=3.0,
                                shrinkA=0, shrinkB=0))
    ax.text(tip1[0] + 0.08, tip1[1], "第 1 列", color=C_RED, fontsize=11,
            ha="left", va="center", fontweight="bold")
    ax.text(tip2[0] - 0.08, tip2[1] - 0.06, "第 2 列", color=C_GREEN, fontsize=11,
            ha="right", va="top", fontweight="bold")

    ax.plot(0, 0, "ks", ms=11, zorder=5)
    ax.text(-0.05, -0.20, "base", fontsize=9.5, ha="center", color="#444444")
    ax.plot(*joint2, "o", color=C_DARK, ms=8, zorder=5)
    ax.text(joint2[0] - 0.10, joint2[1] - 0.22, "肘", fontsize=9.5,
            color="#444444", ha="right")
    ax.plot(*end, "o", color=C_DARK, ms=10, zorder=6)
    ax.text(end[0] + 0.10, end[1] + 0.05, "末端", fontsize=10, color=C_DARK)

    ax.set_xlabel("x（米）")
    ax.set_ylabel("y（米）")
    ax.set_title("左边是几何：两根点线是力臂，两个箭头是速度", fontsize=11.5)

    # ==== 右：计算 ====
    axm.set_xlim(0, 10)
    axm.set_ylim(0, 13.2)
    axm.axis("off")

    lines = [
        (12.6, "输入", C_DARK, 11, "bold"),
        (11.9, f"θ1 = {T1} rad     θ2 = {T2} rad     l1 = l2 = 1", C_DARK, 10, None),
        (10.9, "① 力臂 = 转轴到末端的向量", C_DARK, 11, "bold"),
        (10.2, f"力臂 1（基座→末端）= ({end[0]:.4f}, {end[1]:.4f})", C_RED, 10, None),
        (9.55, f"力臂 2（肘→末端）  = ({arm2[0]:.4f}, {arm2[1]:.4f})", C_GREEN, 10, None),
        (8.6, "② 绕轴转动的速度必然垂直于半径", C_DARK, 11, "bold"),
        (8.0, "   所以每一列 = 力臂逆时针转 90°", C_DARK, 11, "bold"),
        (7.35, "   (x, y)  →  (−y, x)", "#555555", 10, None),
        (6.4, "③ 逐列算", C_DARK, 11, "bold"),
        (5.75, f"第 1 列 = ({-end[1]:.4f}, {end[0]:.4f})", C_RED, 10, None),
        (5.15, f"第 2 列 = ({-arm2[1]:.4f}, {arm2[0]:.4f})", C_GREEN, 10, None),
        (4.35, "④ 两列并排 = 雅可比", C_DARK, 11, "bold"),
    ]
    for y, text, color, size, weight in lines:
        axm.text(0.2, y, text, fontsize=size, color=color,
                 fontweight=weight if weight else "normal", va="center")

    axm.text(2.00, 2.75, "J =", fontsize=12, color=C_DARK, ha="right",
             va="center", family="Consolas")
    y0 = draw_matrix(axm, 3.5, 2.75, J, cell=1.15)
    axm.text(3.5, y0 - 0.55,
             f"|第 1 列| = {np.linalg.norm(v1):.4f} = 力臂 1 的长度",
             fontsize=9.5, color="#555555", ha="center")
    axm.text(3.5, y0 - 0.95,
             f"|第 2 列| = {np.linalg.norm(v2):.4f} = 力臂 2 的长度",
             fontsize=9.5, color="#555555", ha="center")
    axm.text(3.5, y0 - 1.40,
             f"两列 · 各自力臂 = {v1 @ end:.1e} / {v2 @ arm2:.1e} → 正交",
             fontsize=9.5, color="#555555", ha="center")

    fig.suptitle("雅可比的两列 = 两条力臂各自旋转 90°", fontsize=14, y=0.965)

    path = OUT / "W4_D1_1_jacobian_columns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"写出 {path}")
    print(f"末端 {np.round(end, 4).tolist()}   力臂2 {np.round(arm2, 4).tolist()}")
    print(f"第1列 {np.round(v1, 4).tolist()}   第2列 {np.round(v2, 4).tolist()}")
    print(f"|v1| = {np.linalg.norm(v1):.4f}   |v2| = {np.linalg.norm(v2):.4f}")
    print(f"v1·end = {v1 @ end:.2e}   v2·arm2 = {v2 @ arm2:.2e}")
    print("旋转验证:", np.allclose(rot90(end), v1) and np.allclose(rot90(arm2), v2))


if __name__ == "__main__":
    main()
