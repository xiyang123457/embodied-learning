"""平面二连杆正运动学：末端位置、工具尖端、三姿态图。

用途
    把 W3 的两件事落成代码：
    (1) 正运动学，也就是「给定关节角，末端在哪」。这是 W4 求雅可比的直接上游。
    (2) 工具偏移。证明位置和姿态是耦合的：改姿态会连带改工具尖端的位置。
    杆长默认 l1 = l2 = 1，角度一律弧度。

输入输出
    fk(theta1, theta2)            -> (x, y) 末端原点，单位米
    fk_with_tool(theta1, theta2)  -> ((ex, ey), (tx, ty)) 末端原点与工具尖端
    plot(configs, path)           -> 写出一张 PNG，返回 Path

跑完应看到的自检数字（l1 = l2 = 1）
    fk(0, 0)        -> (2.000000,  0.000000)
    fk(pi/2, 0)     -> (0.000000,  2.000000)
    fk(0, pi)       -> (0.000000,  0.000000)    y 实际是 1.22e-16，浮点尾巴
    fk(pi/4, pi/2)  -> (0.000000,  1.414214)
    fk_with_tool(0, pi)   工具尖端 -> (-0.100000, 0.000000)
    fk_with_tool(0, 0)    工具尖端 -> ( 2.100000, 0.000000)

怎么验证跑对了
    python scripts/w3_p3_visualize.py
    控制台四行 fk 与上面逐个对上，且 outputs/figures/ 下出现 W3_P3_1 ~ W3_P3_3 三张图。
"""

import numpy as np

from src.transforms import apply, compose, make_T, rot_z
# apply / make_T / rot_z / compose：工具偏移必须用变换算，不能用位置加法。
#   why 不用 np.array 手拼 4x4：一旦有两处独立实现，哪天 transforms 改了符号，
#       这里不会跟着变，两个文件就悄悄分家了。


# ==== 1. 正运动学 ====
# 第二根杆的绝对方向是 theta1 + theta2，不是 theta2。
# 写成 cos(theta2) 的代码照样能跑、曲线照样光滑，只是形状错了，
# 所以下面的四个自检值里，第四个（两 cos 抵消）是唯一能抓住它的。

def fk(theta1, theta2, l1=1.0, l2=1.0):
    """给定两个关节角（弧度），返回末端原点 (x, y)。"""
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    # l1 / l2：float，杆长，默认 1.0 米
    #   why 允许传参：改杆长能立刻看出「够不到」和「构型变形」两种现象
    # theta1 + theta2：float，第二根杆相对 +x 轴的绝对角
    #   why 一定要相加：theta2 是关节 2 相对杆 1 的转角，不是相对 +x 轴的角
    return float(x), float(y)


def pose(theta1, theta2, l1=1.0, l2=1.0):
    """返回 T_base_end（4x4）。末端姿态角等于 theta1 + theta2。"""
    x, y = fk(theta1, theta2, l1, l2)
    return make_T(rot_z(theta1 + theta2), [x, y, 0.0])
    # rot_z(theta1+theta2)：末端的朝向是两根杆累加的结果，
    #   这就是「末端位姿」里的姿态分量。工具偏移正是靠它把方向带过去。


# ==== 2. 工具偏移 ====
# 工具在末端坐标系里是固定的一段 (0.1, 0)，但它在基座系里的方向随末端转。
# 用 T_base_end @ T_end_tool 算，方向自动跟着走；用位置加法算，方向永远朝 +x。

def fk_with_tool(theta1, theta2, tool_offset=(0.1, 0.0), l1=1.0, l2=1.0):
    """返回 (末端原点, 工具尖端) 两组二维坐标。"""
    T_base_end = pose(theta1, theta2, l1, l2)
    T_end_tool = make_T(np.eye(3), [tool_offset[0], tool_offset[1], 0.0])
    # tool_offset：tuple[float, float]，工具在末端系里的伸出量，默认沿末端 x 轴 0.1 米
    #   why 用 (0.1, 0) 而不是写死：换个偏移量就能看到尖端轨迹整体平移，
    #       说明偏移量本身与姿态无关，变化的只是它的方向

    end = apply(T_base_end, [0.0, 0.0, 0.0])[0]
    # apply(T, [0,0,0]) -> (1,3)，末端系原点在基座系里的位置
    #   为什么直接变换原点：末端坐标系的原点就是末端位置，不必再套一次 fk
    tip = apply(compose(T_base_end, T_end_tool), [0.0, 0.0, 0.0])[0]
    # compose(T_base_end, T_end_tool) -> (4,4)，下标 base←end 接 end←tool，
    #   中间的 end 消掉，结果 base←tool，合法
    #   坑：写成 compose(T_end_tool, T_base_end) 的话下标是 end←tool 接 base←end，
    #       对不上。形状照样合法，数值是另一个东西
    return (float(end[0]), float(end[1])), (float(tip[0]), float(tip[1]))


# ==== 3. 三姿态图 ====
# 三个姿态画在同一个 axes 里，而不是三个子图。
#   why：读者要看的是「折叠后工具尖跑到基座左边」这一件事，
#        三张分开的小图得来回扫眼睛才能对比，画在一起一眼就看见。

def plot(configs, path=None):
    """把若干组 (theta1, theta2) 画在同一张图上，返回输出路径。"""
    import matplotlib.pyplot as plt
    # 延迟导入：只用 fk 的人（比如 W4 的雅可比）不该被迫依赖 matplotlib。

    from pathlib import Path
    path = Path(path or "outputs/figures/W3_P3_1_three_poses.png")

    colors = ["#1f6feb", "#c0392b", "#27ae60"]
    # colors：list[str]，按传入顺序取色。三个姿态三种颜色，
    #   why 固定色表：不同机器重跑时颜色一致，方便和笔记里的图对照

    l1 = l2 = 1.0
    ends = []
    for t1, t2 in configs:
        ends.append(fk(t1, t2, l1, l2))

    # 挑「末端离基座最近」的那个姿态做放大图，也就是折叠姿态。
    #   why：工具只有 0.1 米，而这张图横跨 3.2 米，末端到工具尖的那一段在主图里
    #        只有一个标记点的宽度，箭头都塞不下。必须另给一个按真实比例的放大窗，
    #        否则「姿态」这件事在图上是不可见的。
    zoom_idx = int(np.argmin([np.hypot(x, y) for x, y in ends]))

    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    ax.set_aspect("equal")
    ax.grid(alpha=0.2, linestyle=":")

    for i, (t1, t2) in enumerate(configs):
        color = colors[i % len(colors)]
        joint2 = (l1 * np.cos(t1), l1 * np.sin(t1))
        # joint2：tuple[float, float]，中间关节的位置。画杆 1 的终点要用
        end, tip = fk_with_tool(t1, t2, l1=l1, l2=l2)

        ax.plot([0.0, joint2[0], end[0]], [0.0, joint2[1], end[1]],
                "-o", color=color, lw=3.2, ms=8, solid_capstyle="round",
                label=f"θ=({t1:.2f}, {t2:.2f})")
        # 主图只画「方向」：箭头从末端指向工具尖，末端到尖的那 0.1 米在这里看不见，
        #   真实比例看右上角的放大窗
        ax.annotate("", xy=tip, xytext=end,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                    shrinkA=0, shrinkB=0))
        ax.plot(*tip, marker="o", mfc="white", mec=color, ms=9, mew=1.8)
        # 空心圆＝工具尖端，实心圆＝末端原点。两者必须都出现，
        #   否则「折叠时末端在原点、工具尖在 -0.1」这个现象看不出来

    ax.plot(0, 0, "ks", ms=10)
    # 基座单独用黑色方块标出，它是所有姿态的共同起点
    ax.text(0.03, -0.13, "base", fontsize=9, color="#444444")

    ax.set_xlim(-0.75, 2.75)
    ax.set_ylim(-1.0, 2.3)
    ax.set_xlabel("x（米）")
    ax.set_ylabel("y（米）")
    ax.set_title("平面二连杆：三个姿态的末端原点（实心）与工具尖端（空心）\n"
                 "箭头方向 = 末端的 x 轴方向 = 工具的朝向", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # 放大窗：按真实比例画出折叠姿态的末端与工具，因为 0.1 米在主图里看不见
    ins = ax.inset_axes([0.615, 0.60, 0.355, 0.355])
    t1, t2 = configs[zoom_idx]
    color = colors[zoom_idx % len(colors)]
    end, tip = fk_with_tool(t1, t2, l1=l1, l2=l2)
    ins.set_xlim(-0.215, 0.175)
    ins.set_ylim(-0.125, 0.155)
    ins.set_aspect("equal")
    ins.grid(alpha=0.25, linestyle=":")
    ins.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    ins.axvline(0, color="#dddddd", lw=0.8, zorder=0)

    ins.plot(0, 0, "ks", ms=9, zorder=4)
    ins.plot(*end, "o", color=color, ms=8, zorder=5)
    ins.annotate("", xy=tip, xytext=end,
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4,
                                 shrinkA=0, shrinkB=0))
    ins.plot(*tip, marker="o", mfc="white", mec=color, ms=10, mew=2.0, zorder=5)
    ins.text(0.035, 0.040, "末端＝基座", fontsize=7.5, ha="left", va="center",
             color="#333333")
    ins.text(tip[0], -0.070, f"工具尖 ({tip[0]:.2f}, 0)", fontsize=7.5,
             ha="center", va="center", color=color)
    ins.annotate("", xy=(tip[0], 0.030), xytext=(0.0, 0.030),
                 arrowprops=dict(arrowstyle="<->", color="#666666", lw=1.0))
    ins.text(tip[0] / 2, 0.048, "0.1 m", fontsize=7.0, ha="center", va="bottom",
             color="#666666")
    ins.set_xticks([-0.1, 0.0, 0.1])
    ins.tick_params(labelsize=7)
    ins.set_title("放大：折叠姿态（真实比例）", fontsize=8.5, pad=3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
