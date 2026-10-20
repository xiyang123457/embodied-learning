"""W3 闭卷练习配图：apply 里那个数组一共变形了六次。

用途
    apply 只有三行，但数组在这三行里换了五次形状：
        (N,3) -> (N,4) -> (4,N) -> (4,N) -> (N,4) -> (N,3)
    光看代码很难跟上，画出每一步的实际数字和形状，比读注释快。
    本脚本只画图，不产出交付代码。

输入
    无。用 T = 平移 (1,0,0)，两个点 (0,0,0) 和 (1,2,3)，数字取小值便于核对。

输出
    outputs/figures/W3_apply_pipeline.png

跑完应看到的自检数字
    输入  [[0 0 0], [1 2 3]]
    输出  [[1 0 0], [2 2 3]]        每个点都沿 +x 挪了 1 米
    中途形状  2x3 -> 2x4 -> 4x2 -> 4x2 -> 2x4 -> 2x3
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transforms import apply, make_T

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

T = make_T(np.eye(3), [1.0, 0.0, 0.0])
POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

CELL = 0.42
# CELL：float，矩阵每个格子的边长（数据单位）。4 列时总宽 1.68，3 列时 1.26，
#   在 xlim 0..12.4 里三者不会互相顶到


def draw_matrix(ax, xc, yc, data, title, shape_label, accent="#1f2328"):
    """在 (xc, yc) 居中画一个带方括号的数字矩阵，上方写变量名，下方写形状。"""
    data = np.asarray(data)
    rows, cols = data.shape
    w, h = cols * CELL, rows * CELL
    x0, y0 = xc - w / 2, yc - h / 2

    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x0 + c * CELL, y0 + (rows - 1 - r) * CELL),
                                   CELL, CELL, facecolor="white",
                                   edgecolor="#dcdcdc", lw=0.8))
            ax.text(x0 + (c + 0.5) * CELL, y0 + (rows - 1 - r + 0.5) * CELL,
                    f"{data[r, c]:g}", ha="center", va="center", fontsize=9.5)

    pad = 0.09
    for xb, sgn in ((x0 - pad, 1), (x0 + w + pad, -1)):
        ax.plot([xb, xb], [y0 - pad, y0 + h + pad], color="#666666", lw=1.4)
        ax.plot([xb, xb + 0.10 * sgn], [y0 + h + pad, y0 + h + pad],
                color="#666666", lw=1.4)
        ax.plot([xb, xb + 0.10 * sgn], [y0 - pad, y0 - pad],
                color="#666666", lw=1.4)

    ax.text(xc, y0 + h + 0.30, title, ha="center", va="bottom", fontsize=10.5,
            color=accent, fontweight="bold")
    ax.text(xc, y0 - 0.30, shape_label, ha="center", va="top", fontsize=9.5,
            color="#777777")


def arrow(ax, p1, p2, label):
    """画一条带说明文字的箭头，说明写在箭头中点上方。"""
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                                 color="#8a8a8a", lw=1.6,
                                 shrinkA=0, shrinkB=0))
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    ax.text(mx, my + 0.16, label, ha="center", va="bottom", fontsize=9,
            color="#555555")


def main():
    P = np.atleast_2d(POINTS)
    ones = np.ones((P.shape[0], 1))
    Ph = np.hstack([P, ones])
    PhT = Ph.T
    M = T @ PhT
    back = M.T
    final = back[:, :3]

    fig, ax = plt.subplots(figsize=(12.6, 8.8))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 8.8)
    ax.set_aspect("equal")
    ax.axis("off")

    top_y, bot_y = 6.5, 3.5
    xs = [1.95, 6.20, 10.45]

    draw_matrix(ax, xs[0], top_y, P, "① P", "(2,3)   点的坐标（米）")
    draw_matrix(ax, xs[1], top_y, Ph, "② Ph", "(2,4)   右侧补了一列 1",
                accent="#1f6feb")
    draw_matrix(ax, xs[2], top_y, PhT, "③ Ph.T", "(4,2)   行列互换",
                accent="#1f6feb")

    draw_matrix(ax, xs[2], bot_y, M, "④ T @ Ph.T", "(4,2)   矩阵乘，形状不变")
    draw_matrix(ax, xs[1], bot_y, back, "⑤ (T @ Ph.T).T", "(2,4)   再转回来")
    draw_matrix(ax, xs[0], bot_y, final, "⑥ [: , :3]", "(2,3)   丢掉第 4 列",
                accent="#c0392b")

    arrow(ax, (xs[0] + 0.78, top_y), (xs[1] - 0.99, top_y),
          "np.hstack 加一列 np.ones((2,1))")
    arrow(ax, (xs[1] + 0.99, top_y), (xs[2] - 0.57, top_y), ".T")

    # ③ -> ④ 走外侧弧线，不直上直下。
    #   why：竖线会正好穿过 ③ 下方的形状标注 "(4,2)"，把字割断；
    #        从右侧绕出去，标注区就干净了
    ax.add_patch(FancyArrowPatch((xs[2] + 0.56, top_y), (xs[2] + 0.56, bot_y),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color="#8a8a8a", lw=1.6,
                                 connectionstyle="arc3,rad=-0.45"))
    ax.text(xs[2] + 1.62, (top_y + bot_y) / 2, "左乘 T", ha="center",
            va="center", fontsize=9, color="#555555", rotation=90)

    arrow(ax, (xs[2] - 0.57, bot_y), (xs[1] + 0.99, bot_y), ".T")
    arrow(ax, (xs[1] - 0.99, bot_y), (xs[0] + 0.78, bot_y), "[:, :3]")

    ax.text(6.2, 1.35,
            "输入 (0,0,0) 和 (1,2,3)   →   输出 (1,0,0) 和 (2,2,3)",
            ha="center", fontsize=12, color="#1f2328",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#f7f7f4",
                      edgecolor="#cccccc"))
    ax.text(6.2, 0.55,
            "每个点都沿 +x 挪了 1 米。第 4 维那一列 1 的作用，就是让平移列被原样加进来。",
            ha="center", fontsize=10.5, color="#666666")

    ax.set_title("apply 的三行代码里，数组一共换了五次形状\n"
                 "① → ② 补 1 升维，② → ③ 转置成列向量，⑤ → ⑥ 丢列回三维",
                 fontsize=12.5, pad=14)

    path = OUT / "W3_apply_pipeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"写出 {path}")
    print(f"输入 {P.tolist()}")
    print(f"输出 {apply(T, POINTS).tolist()}")
    print(f"形状 {P.shape} -> {Ph.shape} -> {PhT.shape} -> {M.shape} "
          f"-> {back.shape} -> {final.shape}")
    print("自检:", np.allclose(apply(T, POINTS), final))


if __name__ == "__main__":
    main()
