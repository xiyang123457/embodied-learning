"""W3 · D2 的可视化：四张图，按「原子 → 拼装 → 一个点 → 一张网」的顺序。

用途
    D2 的四个概念（三维旋转、齐次变换的四个分块、坐标系的坐标值、
    四个坐标系之间的边）全部是**方向**问题，文字描述容易看错，画出来一眼就懂。
    本脚本只画图，不产出任何交付代码。

输入
    无。矩阵与坐标全部由 src/transforms.py 现算，不手抄数字。

输出
    outputs/figures/ 下四张 PNG，文件名见 FIGURES 常量：
        W3_D2_1_rot_y_correct_vs_wrong.png   R_y(90°) 正确 vs sin 符号写反
        W3_D2_2_homogeneous_blocks.png       4×4 齐次矩阵的四个分块
        W3_D2_3_same_point_three_frames.png  同一个点在三套坐标系里的三组坐标
        W3_D2_4_frame_tree.png               四个坐标系之间的边与变换名

跑完应看到的自检数字
    W3_D2_3 图上三组坐标分别是 base(1.00, 1.00)、end(0.50, 0.50)、cam(1.00, 0.00)
    四张图全部写出，控制台打印四个文件名
"""

import sys
from pathlib import Path

import matplotlib
# matplotlib：画图。这里只用 Agg 后端，因为目标产物是 PNG 文件，不需要交互窗口。
#   坑：不显式设后端时，在某些无显示环境（服务器、CI）里 import pyplot 会直接报错。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 把仓库根目录插到 sys.path 最前面。
#   为什么需要：直接跑 `python scripts/xxx.py` 时，sys.path[0] 是 scripts/ 而不是仓库根，
#   `import src.transforms` 会抛 ModuleNotFoundError。
#   为什么不用 `python -m scripts.xxx`：README 里全仓库统一是 `python scripts/xxx.py`，
#   这里跟着统一，读者不用记两种运行方式。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transforms import apply, inverse, make_T, rot_y, rot_z

# 中文字体：Windows 自带雅黑。不设的话所有中文会变成方框。
#   unicode_minus 关掉，否则负号会渲染成方框，坐标轴上的 -1 变成豆腐块。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("outputs/figures")
# OUT：Path，输出目录。相对路径，必须在仓库根目录运行，否则会建到脚本所在目录下。
OUT.mkdir(parents=True, exist_ok=True)

FIGURES = [
    "W3_D2_1_rot_y_correct_vs_wrong.png",
    "W3_D2_2_homogeneous_blocks.png",
    "W3_D2_3_same_point_three_frames.png",
    "W3_D2_4_frame_tree.png",
]

AXIS_COLORS = {"x": "#c0392b", "y": "#27ae60", "z": "#2c6fbb"}
# AXIS_COLORS：dict，坐标轴颜色。红绿蓝各对应 xyz，
#   为什么用三色而不是统一颜色：图上同时有原始轴和旋转后的轴，靠颜色区分才不用数箭头。


# ==== 1. R_y 的符号：正确与写反并排 ====
# 这两张必须放在同一张图里才有意义：单独看「正确的图」你看不出错版长什么样，
# 单独看「错版」你也发现不了它是错的，只有并排才能看出 z 轴被送到了相反方向。

def draw_axes_3d(ax, R, title):
    """在 3D 轴上画「原始坐标系（灰）+ 经 R 变换后的坐标系（彩色）」。"""
    origin = np.zeros(3)
    for i, name in enumerate("xyz"):
        e = np.zeros(3)
        e[i] = 1.0
        ax.quiver(*origin, *e, color="#8d949c", linewidth=1.7,
                  arrow_length_ratio=0.15, alpha=0.6)
        # 参考轴不标 x/y/z 字母。
        #   why：R_y 会让 y' 与原 y 几乎重合，两套字母在 3D 投影下必然叠在一起，
        #        出现 "y y'" 这种糊成一团的效果。字母只留给彩色轴，
        #        灰色轴的身份靠颜色和图注说明，读起来反而干净

    for i, name in enumerate("xyz"):
        e = np.zeros(3)
        e[i] = 1.0
        # 旋转后的轴画到 0.92，比灰色参考轴（1.0）短一点。
        #   why：R_y 会让 y 轴原地不动、z 轴转到 x 轴，彩色箭头会正好盖住灰色参考箭头，
        #        短一截之后参考轴总能在箭头尖端露出来，读者才看得出「转了多少」
        v = R @ e * 0.92
        ax.quiver(*origin, *v, color=AXIS_COLORS[name], linewidth=2.8,
                  arrow_length_ratio=0.16)
        ax.text(*(v * 1.42), f"{name}'", color=AXIS_COLORS[name],
                fontsize=12, fontweight="bold", ha="center", va="center")

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-1.15, 1.15)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-62)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title, fontsize=12.5, pad=2)


def draw_matrix(ax, rows, colors, result, result_color):
    """把一个 3×3 矩阵按格子画出来，逐格上色，下方写结果。

    rows 是 3 行 3 列的字符串，colors 是同形状的颜色。
    为什么不用一段等宽文本：逐格画才能单独给「出问题的那两个 sin」上色，
    等宽文本只能整块一个颜色，读者还得自己找哪一处不一样。
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    xs = [2.75, 4.80, 6.85]
    # xs：list[float]，三列的中心横坐标。间距 2.05 是为了让 "-sinθ" 这种 5 字符
    #   的格子不互相顶在一起
    ys = [5.05, 3.95, 2.85]
    # ys：list[float]，三行的中心纵坐标，行距 1.10

    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            ax.text(xs[c], ys[r], cell, ha="center", va="center",
                    fontsize=14, family="Consolas", color=colors[r][c])

    for xb, sgn in ((1.45, 1), (8.15, -1)):
        # sgn：int，左括号的钩子朝右（+1），右括号的钩子朝左（-1）
        ax.plot([xb, xb], [1.95, 5.95], color="#555555", lw=1.8)
        ax.plot([xb, xb + 0.24 * sgn], [5.95, 5.95], color="#555555", lw=1.8)
        ax.plot([xb, xb + 0.24 * sgn], [1.95, 1.95], color="#555555", lw=1.8)

    ax.text(4.80, 1.25, result, ha="center", fontsize=12,
            color=result_color, family="Consolas")


def fig_rot_y_check():
    """画 R_y(90°) 的正确版与符号写反版：上半是几何，下半是矩阵写法。"""
    R_good = rot_y(np.pi / 2)
    # R_good：ndarray (3,3)，[[0,0,1],[0,1,0],[-1,0,0]]
    #   why：直接用库里那份，保证上半的几何图和 src/transforms.py 是同一个东西
    R_bad = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    # R_bad：ndarray (3,3)，把 R_y 里 s 和 -s 的位置对调后的结果
    #   why 手写而不用 rot_y(-pi/2)：这一版要展示"写错的矩阵长什么样"，
    #       数学上它等于 R_y(-90°)，所以它照样是合法旋转，det 仍是 +1

    DARK, HL_GOOD, HL_BAD = "#1f2328", "#2c6fbb", "#c0392b"
    rows_good = [["cosθ", "0", "sinθ"], ["0", "1", "0"], ["-sinθ", "0", "cosθ"]]
    rows_bad = [["cosθ", "0", "-sinθ"], ["0", "1", "0"], ["sinθ", "0", "cosθ"]]
    # rows_*：list[list[str]]，3×3 的符号矩阵
    #   why 用符号而不是 90° 的数值：符号版才看得出「写法」哪里不同；
    #       数值版两行都是 0 1 -1，差别被 0 淹没了，反而看不出问题
    colors_good = [[DARK, DARK, HL_GOOD], [DARK, DARK, DARK], [HL_GOOD, DARK, DARK]]
    colors_bad = [[DARK, DARK, HL_BAD], [DARK, DARK, DARK], [HL_BAD, DARK, DARK]]
    # 上色的两格就是唯一不同的两处：[0][2] 和 [2][0]

    fig = plt.figure(figsize=(11.8, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.15, 1.35], hspace=0.02, wspace=0.04)

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    m1 = fig.add_subplot(gs[1, 0])
    m2 = fig.add_subplot(gs[1, 1])

    draw_axes_3d(ax1, R_good, "正确写法")
    draw_axes_3d(ax2, R_bad, "sin 符号写反")

    draw_matrix(m1, rows_good, colors_good,
                "θ=90°  →  R·(0,0,1) = ( 1, 0, 0)", HL_GOOD)
    draw_matrix(m2, rows_bad, colors_bad,
                "θ=90°  →  R·(0,0,1) = (-1, 0, 0)", HL_BAD)

    fig.text(0.5, 0.372, "两处 sin 的位置对调，其余元素完全相同",
             ha="center", fontsize=11, color="#666666")
    fig.suptitle("R_y(θ)：正确写法与错误写法的对照\n"
                 "上半灰色三根为原坐标系 x/y/z，彩色三根为旋转后 x'/y'/z'（红=x 绿=y 蓝=z）",
                 fontsize=12.5, y=0.985, linespacing=1.6)

    path = OUT / FIGURES[0]
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ==== 2. 4×4 齐次矩阵的四个分块 ====
# 新手看着一个 4×4 数字矩阵，最容易问"这四个数到底是干嘛的"。
# 分块着色比任何文字都快：左上 3×3 管方向，右侧一列管位置，最后一行只是个占位。

def fig_homogeneous_blocks():
    """把 T_base_cam 的 16 个数字画成格，并按分块着色标注。"""
    T = make_T(rot_z(np.pi / 2), [1.0, 0.0, 0.0])
    # T：ndarray (4,4)，[[0,-1,0,1],[1,0,0,0],[0,0,1,0],[0,0,0,1]]
    #   why 选这个矩阵：它的元素只有 0、1、-1，读者能一眼核对，
    #       不会被一串小数干扰注意力

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    ax.set_xlim(-0.6, 5.4)
    ax.set_ylim(-2.6, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(4):
        for c in range(4):
            y = 4 - r
            if c < 3 and r < 3:
                face, edge = "#e8f1fb", "#2c6fbb"      # 旋转块
            elif c == 3 and r < 3:
                face, edge = "#fdeee2", "#d97706"      # 平移列
            elif r == 3:
                face, edge = "#eeeeee", "#7a7a7a"      # 最后一行
            else:
                face, edge = "#ffffff", "#cccccc"
            ax.add_patch(plt.Rectangle((c, y - 1), 1, 1, facecolor=face,
                                       edgecolor=edge, linewidth=1.6))
            val = T[r, c]
            # 小于 1e-12 的当成 0：cos(90°) 是 6.12e-17，直接打印会把格子撑成
            #   "6.12323e-17"，读者反而看不到「这一格就是 0」这个关键信息
            val = 0.0 if abs(val) < 1e-12 else val
            txt = f"{val:g}"
            ax.text(c + 0.5, y - 0.5, txt, ha="center", va="center",
                    fontsize=19, color="#1f2328",
                    fontweight="bold" if val != 0 else "normal")

    ax.annotate("旋转块 R（3×3）\n只管方向", xy=(1.5, 4.0), xytext=(1.5, 5.25),
                ha="center", fontsize=11, color="#2c6fbb",
                arrowprops=dict(arrowstyle="-|>", color="#2c6fbb", lw=1.6))
    ax.annotate("平移列 t（3×1）\n只管位置", xy=(3.5, 4.0), xytext=(4.55, 4.75),
                ha="center", fontsize=11, color="#d97706",
                arrowprops=dict(arrowstyle="-|>", color="#d97706", lw=1.6))
    ax.annotate("最后一行永远 [0,0,0,1]\n第 4 维的占位符", xy=(2.0, 0.0), xytext=(2.0, -1.28),
                ha="center", fontsize=11, color="#7a7a7a",
                arrowprops=dict(arrowstyle="-|>", color="#7a7a7a", lw=1.6))

    ax.text(2.0, 6.15, "T_base_cam = 绕 z 转 90° + 平移 (1, 0, 0)",
            ha="center", fontsize=13)
    ax.text(2.4, -2.20,
            "同一个矩阵，读法只有一种：把 cam 系里的点变到 base 系。\n"
            "读反了就得到另一个变换，而程序不会报错。",
            ha="center", fontsize=10.5, color="#444444")

    path = OUT / FIGURES[1]
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ==== 3. 同一个点，三套坐标 ====
# 这张图回答 D2 最核心的问题：物理上只有一个点，为什么有三组数字？
# 三组数字都对，只是各自的参照系不同。

def fig_same_point_three_frames():
    """画 base / end / cam 三个坐标系，同一个目标点在三者下的坐标。"""
    p_base = np.array([1.0, 1.0, 0.0])
    # p_base：ndarray (3,)，目标点在 base 系里的位置
    #   why 取 (1,1,0)：整数，读图时能立刻和坐标标注对上，不需要换算

    T_base_end = make_T(np.eye(3), [0.5, 0.5, 0.0])
    T_base_cam = make_T(rot_z(np.pi / 2), [1.0, 0.0, 0.0])

    # 用逆变换反推同一个点在各系里的局部坐标，不手算，避免图和代码脱节
    p_end = apply(inverse(T_base_end), p_base)[0]
    p_cam = apply(inverse(T_base_cam), p_base)[0]

    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    ax.set_xlim(-0.45, 2.0)
    ax.set_ylim(-0.85, 1.9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.18, linestyle=":")
    ax.axhline(0, color="#d0d0d0", lw=0.8, zorder=0)
    ax.axvline(0, color="#d0d0d0", lw=0.8, zorder=0)

    def draw_frame(origin, angle_deg, color, name, scale=0.32,
                   off=((0.0, -0.16), (-0.13, 0.03))):
        """在 origin 处画一组带名字的坐标轴。角度是绕 z 的逆时针度数。

        off 是 x、y 两个轴标签相对箭头端点的额外偏移。
        为什么要单独给偏移：base 的 x 轴和 cam 的 y 轴都落在 y=0 这条线上，
        不分开挪，两个标签会叠成 "x_base/y_cam"。
        """
        a = np.deg2rad(angle_deg)
        for i, lab in enumerate("xy"):
            vec = np.array([np.cos(a + i * np.pi / 2), np.sin(a + i * np.pi / 2)]) * scale
            ax.annotate("", xy=origin + vec, xytext=origin,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))
            p = origin + vec * 1.30 + np.array(off[i])
            ax.text(*p, f"{lab}_{name}", color=color, fontsize=9.5,
                    ha="center", va="center")

    draw_frame(np.array([0.0, 0.0]), 0, "#1f2328", "base")
    draw_frame(np.array([0.5, 0.5]), 0, "#2c6fbb", "end")
    draw_frame(np.array([1.0, 0.0]), 90, "#27ae60", "cam",
               off=((0.11, 0.08), (0.02, 0.17)))

    ax.scatter(*p_base[:2], marker="*", s=460, color="#c0392b", zorder=5,
               edgecolor="white", linewidth=1.2)
    ax.text(p_base[0] + 0.07, p_base[1] + 0.07, "目标点", color="#c0392b", fontsize=12)

    rows = [("base 系", f"({p_base[0]:.2f}, {p_base[1]:.2f})", "#1f2328"),
            ("end 系", f"({p_end[0]:.2f}, {p_end[1]:.2f})", "#2c6fbb"),
            ("cam 系", f"({p_cam[0]:.2f}, {p_cam[1]:.2f})", "#27ae60")]
    for i, (name, coord, color) in enumerate(rows):
        ax.text(1.28, 1.72 - i * 0.17, f"{name}：{coord}", color=color,
                fontsize=11.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=color, alpha=0.9))

    ax.set_title("同一个目标点，三套坐标系，三组坐标", fontsize=13, pad=12)
    ax.set_xlabel("base 系 x（米）")
    ax.set_ylabel("base 系 y（米）")

    path = OUT / FIGURES[2]
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ==== 4. 四个坐标系之间的边 ====
# 这张图是 D2 的交付物之一（计划书 B3 要求手画）。
# 每条边都写成 T_target_source，箭头方向就是变换方向。

def fig_frame_tree():
    """画 world/base/end/tool/cam/obj 六个坐标系与它们之间的变换边。"""
    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    W, H = 1.45, 1.0
    # W / H：float，框的宽和高（数据单位，1 单位约等于图上 1 英寸）
    #   why 定成 1.45×1.0：上下两排要放下 4 个框，间隙还要容得下 "T_base_world"
    #       这种 12 字符的标签，间隙小于 1.4 个字符就挤了
    boxes = {
        "world": (0.35, 5.40, "#f2f2f2", "#7a7a7a"),
        "base":  (3.35, 5.40, "#e8f1fb", "#2c6fbb"),
        "end":   (6.35, 5.40, "#e8f1fb", "#2c6fbb"),
        "tool":  (9.35, 5.40, "#e8f1fb", "#2c6fbb"),
        "cam":   (6.35, 2.30, "#e7f7ef", "#27ae60"),
        "obj":   (9.35, 2.30, "#fdeee2", "#d97706"),
    }
    # boxes：dict，key 是系名，value 是 (左下角 x, 左下角 y, 填充色, 边色)
    #   why 手排固定坐标而不用自动布局：六个框是一棵树，自动布局容易让边交叉，
    #       手排能保证「什么由什么决定」这一层信息落在不重叠的位置

    for name, (x, y, face, edge) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), W, H,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, linewidth=2.0))
        ax.text(x + W / 2, y + H / 2, name, ha="center", va="center",
                fontsize=13, color=edge, fontweight="bold")

    edges = [
        ("T_base_world", "恒等", "#7a7a7a", (1.80, 5.90), (3.35, 5.90), (2.57, 6.22), (2.57, 5.55)),
        ("T_base_end", "由关节角决定", "#2c6fbb", (4.80, 5.90), (6.35, 5.90), (5.57, 6.22), (5.57, 5.55)),
        ("T_end_tool", "固定值", "#2c6fbb", (7.80, 5.90), (9.35, 5.90), (8.57, 6.22), (8.57, 5.55)),
        ("T_base_cam", "手眼标定", "#27ae60", (4.05, 5.35), (7.05, 3.35), (4.45, 4.55), (4.45, 4.22)),
        ("T_cam_obj", "相机测出", "#d97706", (7.80, 2.80), (9.35, 2.80), (8.57, 3.12), (8.57, 2.45)),
    ]
    # edges：list[tuple]，元素依次是
    #   (变换名, 由什么决定, 颜色, 箭头起点, 箭头终点, 名字坐标, 说明坐标)
    #   why 显式给起点终点坐标：从框中心连线会让箭头压在框上、标签落进框里，
    #       只连「出框右侧到入框左侧」的边，间隙才是干净的标注区
    #   why 标签坐标也写死：水平边的名字放线上方、说明放线下方；对角边整体挪到线的左侧，
    #       否则 "T_base_cam" 会被斜线穿过，读起来像被划掉了
    #   why 把"由什么决定"写进图里：这是 B3 要求的第三件事，
    #       不写的话这张图只是一张连接图，说明不了哪些量已知、哪些要求

    for tname, note, color, p1, p2, name_xy, note_xy in edges:
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18,
                                     color=color, linewidth=2.0, shrinkA=3, shrinkB=3))
        ax.text(*name_xy, tname, ha="center", fontsize=10.5, color=color,
                fontweight="bold")
        ax.text(*note_xy, note, ha="center", fontsize=9, color="#8a8a8a")

    ax.text(5.60, 1.55, "求  T_base_obj = T_base_cam @ T_cam_obj",
            ha="center", fontsize=12.5, color="#1f2328",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f4", edgecolor="#cccccc"))
    ax.text(5.60, 0.85, "求  T_end_obj = inverse(T_base_end) @ T_base_obj",
            ha="center", fontsize=12.5, color="#1f2328",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f4", edgecolor="#cccccc"))
    ax.text(5.60, 0.25, "两条链都必须「下标首尾相接」才合法。"
                        "world→base 在实际系统里通常取恒等，因为 base 就是根坐标系。",
            ha="center", fontsize=10, color="#666666")

    ax.set_title("四个坐标系：箭头方向就是变换方向", fontsize=14, pad=12)

    path = OUT / FIGURES[3]
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ==== 5. 主流程 ====

def main():
    paths = [fig_rot_y_check(), fig_homogeneous_blocks(),
             fig_same_point_three_frames(), fig_frame_tree()]
    for p in paths:
        print(f"写出 {p}")

    # 自检：第三张图上的三组坐标必须是这三个值，否则图和代码已经脱节
    p_base = np.array([1.0, 1.0, 0.0])
    p_end = apply(inverse(make_T(np.eye(3), [0.5, 0.5, 0.0])), p_base)[0]
    p_cam = apply(inverse(make_T(rot_z(np.pi / 2), [1.0, 0.0, 0.0])), p_base)[0]
    print(f"自检 base={np.round(p_base[:2], 2)} end={np.round(p_end[:2], 2)} "
          f"cam={np.round(p_cam[:2], 2)}")


if __name__ == "__main__":
    main()
