"""W3 · P3 的可视化：三张图，每张只讲一件事。

用途
    1. 三姿态总览（计划书 P3 的交付图），由 src/two_link.py 的 plot 直接产出
    2. 工具为什么在完全折叠时指向 -x，而不是 +x
    3. 第二根杆的角度写成 theta2 而不是 theta1+theta2 会错成什么样

输入
    无。杆长取 l1 = l2 = 1，角度一律弧度。

输出
    outputs/figures/ 下三张 PNG：
        W3_P3_1_three_poses.png        伸直 / 折叠 / 肘 90° 三个姿态
        W3_P3_2_same_position.png      同一末端位置、两种构型，工具尖分叉
        W3_P3_3_angle_sum_error.png    正确角度累加与错误写法的末端轨迹

跑完应看到的自检数字
    fk(0, 0)       = (2.000000, 0.000000)
    fk(pi/2, 0)    = (0.000000, 2.000000)
    fk(0, pi)      = (0.000000, 0.000000)
    fk(pi/4, pi/2) = (0.000000, 1.414214)
    折叠时工具尖端 = (-0.100000, 0.000000)
    错误写法的轨迹半径 = 1.0，正确写法 = 1.414214
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 把仓库根目录放进 sys.path，否则 `python scripts/xxx.py` 找不到 src 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.two_link import fk, fk_with_tool, plot as plot_poses

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

L1 = L2 = 1.0


# ==== 1. 工具朝向：为什么折叠时尖端在 -0.1 ====
# 工具在末端坐标系里永远是 (0.1, 0)，问题在于「末端的 x 轴」指向哪。
# 伸直时它朝 +x，折叠时末端转了 180°，它朝 -x，工具尖于是跑到基座左边。

def fig_same_position_two_configs():
    """同一个末端位置、两种构型：姿态不同，工具尖也就落在不同地方。

    选这两组角是有讲究的：(θ1, θ2) 与 (θ1+θ2, -θ2) 在 l1 = l2 时给出同一个末端位置，
    但末端朝向分别是 θ1+θ2 和 θ1，不相等。于是位置完全重合，工具尖却分开了。
    这正是逆运动学「多解」的最直观样子，不用写求解器也能画出来。
    """
    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 1.85)
    ax.set_ylim(-0.35, 1.85)
    ax.grid(alpha=0.2, linestyle=":")
    ax.axhline(0, color="#e5e5e5", lw=0.9, zorder=0)
    ax.axvline(0, color="#e5e5e5", lw=0.9, zorder=0)

    cases = [
        ((0.0, np.pi / 2), "#1f6feb", "构型 A  θ=(0, π/2)", (1.02, 1.26)),
        ((np.pi / 2, -np.pi / 2), "#c0392b", "构型 B  θ=(π/2, -π/2)", (1.14, 0.72)),
    ]
    # 第三个元素是工具尖端标签的位置，两组分开摆，避免叠在同一个角落

    for (t1, t2), color, label, lab_xy in cases:
        j2 = (L1 * np.cos(t1), L1 * np.sin(t1))
        end, tip = fk_with_tool(t1, t2, l1=L1, l2=L2)
        ax.plot([0, j2[0], end[0]], [0, j2[1], end[1]], "-o",
                color=color, lw=3.4, ms=8, label=label, solid_capstyle="round")
        ax.annotate("", xy=tip, xytext=end,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6,
                                    shrinkA=0, shrinkB=0))
        ax.plot(*tip, marker="o", mfc="white", mec=color, ms=13, mew=2.4)
        ax.annotate(f"工具尖 ({tip[0]:.2f}, {tip[1]:.2f})", xy=tip, xytext=lab_xy,
                    fontsize=10, color=color, ha="left", va="center")

    ax.plot(1.0, 1.0, marker="*", ms=24, color="#111111", zorder=6)
    ax.annotate("两个构型的末端重合在 (1.00, 1.00)", xy=(1.0, 1.0),
                xytext=(0.12, 0.52), fontsize=10, color="#111111",
                arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.4,
                                connectionstyle="arc3,rad=-0.2"))

    ax.plot(0, 0, "ks", ms=11)
    ax.text(-0.06, -0.16, "base", fontsize=9.5, ha="center", color="#444444")
    ax.set_xlabel("x（米）")
    ax.set_ylabel("y（米）")
    ax.set_title("末端位置相同，姿态不同 → 工具尖落在两个地方\n"
                 "(θ1, θ2) 与 (θ1+θ2, −θ2) 在 l1 = l2 时末端重合")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92)

    path = OUT / "W3_P3_2_same_position.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ==== 2. 角度累加：theta1+theta2 与 theta2 的区别 ====
# 固定 theta2 = pi/2，让 theta1 转一整圈，看末端画出的轨迹。
# 正确写法画出一个半径 sqrt(2) 的圆，错误写法画出一个半径 1、圆心在 (0,1) 的圆。
# 两条曲线都光滑、都闭合，光看代码看不出谁对，只有画出来才露馅。

def fig_angle_sum_error():
    """扫 theta1 一整圈，对比正确与错误的角度累加。"""
    t2 = np.pi / 2
    ts = np.linspace(0, 2 * np.pi, 400)

    good_x = np.cos(ts) + np.cos(ts + t2)
    good_y = np.sin(ts) + np.sin(ts + t2)
    bad_x = np.cos(ts) + np.cos(t2)
    bad_y = np.sin(ts) + np.sin(t2)

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.set_aspect("equal")
    ax.grid(alpha=0.2, linestyle=":")
    ax.axhline(0, color="#e5e5e5", lw=0.9, zorder=0)
    ax.axvline(0, color="#e5e5e5", lw=0.9, zorder=0)

    ax.plot(good_x, good_y, color="#1f6feb", lw=3.0,
            label="正确  cos(θ1+θ2)：半径 √2 ≈ 1.414")
    ax.plot(bad_x, bad_y, color="#c0392b", lw=2.4, linestyle="--",
            label="错误  cos(θ2)：半径 1，圆心 (0, 1)")

    ax.scatter([0], [0], marker="s", s=90, color="black", zorder=5)
    ax.annotate("基座", xy=(0, 0), xytext=(-0.05, -0.22), fontsize=9.5,
                ha="right", color="#444444")

    # 红虚圆的圆心，单独标出来：它的半径是从这里量的，不是从基座量的
    ax.plot([0], [1], marker="+", ms=11, mew=2.0, color="#c0392b", zorder=5)
    ax.annotate("错误曲线的圆心 (0, 1)", xy=(0, 1), xytext=(0.18, 0.72),
                fontsize=9.5, color="#c0392b",
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.2))

    # 两条半径都从各自的圆心量到圆周，长度才等于标注值
    ax.annotate("", xy=(np.sqrt(2), 0.10), xytext=(0, 0.10),
                arrowprops=dict(arrowstyle="<->", color="#1f6feb", lw=1.5))
    ax.text(np.sqrt(2) / 2, 0.18, "1.414", color="#1f6feb", fontsize=10, ha="center")
    ax.annotate("", xy=(0.30, 1.0), xytext=(0.30, 2.0),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.5))
    ax.text(0.38, 1.50, "1.0", color="#c0392b", fontsize=10, ha="left")

    # 两条曲线的交点：θ1=0 时两个公式给出同一个点，这是最容易误判的抽查位置。
    #   说明文字放到图底部空白处，不画箭头：箭头会横穿两条曲线，反而更乱。
    ax.plot([1.0], [1.0], marker="o", ms=13, mfc="none", mec="#6b2d9e", mew=2.2,
            zorder=6)
    ax.text(0.0, -2.20, "紫色圈：θ1=0 时两个公式都给 (1, 1)。只抽查这一个点会误判通过。",
            ha="center", fontsize=10, color="#6b2d9e")

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.45, 2.4)
    ax.set_xlabel("x（米）")
    ax.set_ylabel("y（米）")
    ax.set_title("固定肘角 θ2 = 90° 不动，只转肩角 θ1 一整圈：末端画出这个圆\n"
                 "圆的半径由 θ2 决定，与 θ1 无关；两条曲线的半径对不上")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92)

    path = OUT / "W3_P3_3_angle_sum_error.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ==== 3. 两个角分别是什么、半径为什么由 θ2 决定 ====
# 图 3 的两条曲线半径不同，根子在这里：
#   θ1 是「整个手臂朝哪」，转它等于把整条手臂当刚体绕基座转，末端只能沿圆周走；
#   θ2 是「肘弯多少」，它决定末端离基座多远，也就是那个圆的半径。
# 把这两个作用分开画，图 3 的半径问题就不需要解释了。

def fig_theta_definition():
    """画一组姿态，标出 θ1、θ2 的定义，以及末端轨迹圆的半径来源。"""
    from matplotlib.patches import Arc

    t1 = np.deg2rad(45)
    t2 = np.pi / 2
    j2 = (L1 * np.cos(t1), L1 * np.sin(t1))
    end = fk(t1, t2, L1, L2)
    ext = (j2[0] + 0.66 * np.cos(t1), j2[1] + 0.66 * np.sin(t1))
    # ext：杆 1 的延长线终点。θ2 要从这条延长线量起，不是从 +x 轴量起

    R = float(np.hypot(end[0], end[1]))
    # R：float，末端到基座的距离，也就是「整条手臂伸直后的等效长度」
    #   R = sqrt(l1^2 + l2^2 + 2*l1*l2*cos(theta2))，theta2=90° 时为 sqrt(2)

    fig, ax = plt.subplots(figsize=(9.0, 9.8))
    fig.subplots_adjust(bottom=0.30)
    ax.set_aspect("equal")
    ax.set_xlim(-1.68, 1.98)
    ax.set_ylim(-1.58, 1.78)
    ax.grid(alpha=0.16, linestyle=":")
    ax.axhline(0, color="#e0e0e0", lw=0.9, zorder=0)
    ax.axvline(0, color="#e0e0e0", lw=0.9, zorder=0)

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R * np.cos(th), R * np.sin(th), ":", color="#b0b0b0", lw=1.7, zorder=1)
    ax.text(-1.60, -1.44, f"灰色圆 = 末端能走到的轨迹，半径 {R:.3f}",
            fontsize=9.5, color="#8a8a8a", ha="left", va="top")

    ax.plot([0.0, 0.62], [0.0, 0.0], ":", color="#9aa0a6", lw=1.3)
    ax.text(0.64, -0.05, "+x 参考方向", fontsize=9, color="#8a8a8a", va="top")

    ax.add_patch(Arc((0, 0), 1.0, 1.0, theta1=0, theta2=45,
                     color="#c0392b", lw=2.0))
    ax.text(0.56, 0.14, "θ1", color="#c0392b", fontsize=15, fontweight="bold")

    ax.plot([j2[0], ext[0]], [j2[1], ext[1]], "--", color="#9aa0a6", lw=1.5)
    ax.text(ext[0] + 0.03, ext[1] + 0.05, "杆 1 的延长线", fontsize=9,
            color="#8a8a8a", va="bottom")

    ax.add_patch(Arc(j2, 0.86, 0.86, theta1=45, theta2=135,
                     color="#27ae60", lw=2.0))
    ax.text(j2[0] - 0.40, j2[1] + 0.36, "θ2", color="#27ae60",
            fontsize=15, fontweight="bold")

    ax.plot([0.0, j2[0], end[0]], [0.0, j2[1], end[1]], "-o",
            color="#1f6feb", lw=3.6, ms=9, solid_capstyle="round", zorder=4)
    ax.plot([0.0, end[0]], [0.0, end[1]], ":", color="#7b2d9e", lw=2.2, zorder=3)
    ax.text(0.10, 0.60, f"R = {R:.3f}", color="#7b2d9e", fontsize=11,
            rotation=90, va="bottom", ha="left")

    ax.plot(0, 0, "ks", ms=11, zorder=5)
    ax.text(-0.07, -0.20, "base", fontsize=9.5, ha="center", color="#444444")
    ax.plot(*end, marker="o", mfc="white", mec="#1f6feb", ms=13, mew=2.4, zorder=5)

    ax.set_xlabel("x（米）")
    ax.set_ylabel("y（米）")
    ax.set_title("θ1 管方向，θ2 管半径", fontsize=13.5, pad=10)

    note = (
        "θ1（肩）：从 +x 轴量起。转它 = 整条手臂绕基座转，末端只沿灰色圆移动。\n"
        "θ2（肘）：从杆 1 延长线量起。转它 = 改肘的弯曲程度，它决定圆的半径。\n"
        "R = √(l1² + l2² + 2·l1·l2·cos θ2)\n"
        "l1 = l2 = 1：θ2 = 0 → R = 2；θ2 = 90° → R = 1.414；θ2 = 180° → R = 0"
    )
    fig.text(0.5, 0.13, note, ha="center", va="center", fontsize=10.2,
             color="#333333", linespacing=1.85,
             bbox=dict(boxstyle="round,pad=0.7", facecolor="#fbfbf8",
                       edgecolor="#cccccc"))

    path = OUT / "W3_P3_4_theta_and_radius.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ==== 4. 主流程 ====

def main():
    poses = [(0.0, 0.0), (0.0, np.pi), (np.pi / 4, np.pi / 2)]
    p1 = plot_poses(poses, OUT / "W3_P3_1_three_poses.png")
    p2 = fig_same_position_two_configs()
    p3 = fig_angle_sum_error()
    p4 = fig_theta_definition()

    for p in (p1, p2, p3, p4):
        print(f"写出 {p}")

    print("\n== fk 自检 ==")
    for t1, t2 in poses + [(np.pi / 2, 0.0)]:
        x, y = fk(t1, t2, L1, L2)
        print(f"  fk({t1:.4f}, {t2:.4f}) = ({x:.6f}, {y:.6f})")

    end, tip = fk_with_tool(0.0, np.pi, l1=L1, l2=L2)
    print(f"\n折叠时：末端原点 {np.round(end, 6)}，工具尖端 {np.round(tip, 6)}")
    print(f"错误写法的轨迹半径 {1.0:.6f}，正确写法 {np.sqrt(2):.6f}")


if __name__ == "__main__":
    main()
