"""W3 · P4 的配图：把「相机 -> 基座 -> 末端」这条乘法链画成一条路径。

用途
    scripts/camera_to_end.py 打印的是三组数字，图上要回答的是另一件事：
    这三组数字分别是从哪条路走出来的。四个坐标系画在一张俯视图上，
    每条边标出它对应哪个 T，再标出目标在三套坐标系下的不同读数。

输入
    无。场景与 camera_to_end.py 完全一致。

输出
    outputs/figures/W3_P4_1_chain.png

跑完应看到的自检数字
    目标在 base 系 (1.00, 1.00)
    目标在 cam  系 (1.00, 0.00)
    目标在 end  系 (0.50, 0.50)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transforms import apply, compose, inverse, make_T, rot_z

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

CAM_POS, CAM_YAW = [1.0, 0.0, 0.0], np.pi / 2
END_POS, END_YAW = [0.5, 0.5, 0.0], 0.0
OBJ_IN_CAM = [1.0, 0.0, 0.0]


def draw_frame(ax, origin, yaw_deg, color, name, scale=0.30, lab_off=(0.0, -0.15)):
    """在 origin 处画一组带名字的坐标轴。yaw_deg 是绕 z 的逆时针度数。"""
    a = np.deg2rad(yaw_deg)
    for i, lab in enumerate("xy"):
        vec = np.array([np.cos(a + i * np.pi / 2), np.sin(a + i * np.pi / 2)]) * scale
        ax.annotate("", xy=origin + vec, xytext=origin,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))
        p = origin + vec * 1.34 + np.array(lab_off if i == 0 else (-0.14, 0.03))
        ax.text(*p, f"{lab}_{name}", color=color, fontsize=9.5, ha="center",
                va="center")


def main():
    T_base_cam = make_T(rot_z(CAM_YAW), CAM_POS)
    T_base_end = make_T(rot_z(END_YAW), END_POS)
    T_cam_obj = make_T(np.eye(3), OBJ_IN_CAM)

    T_base_obj = compose(T_base_cam, T_cam_obj)
    T_end_obj = compose(inverse(T_base_end), T_base_obj)

    p_base = apply(T_base_obj, [0.0, 0.0, 0.0])[0][:2]
    p_cam = apply(T_cam_obj, [0.0, 0.0, 0.0])[0][:2]
    p_end = apply(T_end_obj, [0.0, 0.0, 0.0])[0][:2]

    fig, ax = plt.subplots(figsize=(9.6, 7.8))
    ax.set_aspect("equal")
    ax.set_xlim(-0.45, 1.95)
    ax.set_ylim(-1.05, 1.85)
    ax.grid(alpha=0.16, linestyle=":")
    ax.axhline(0, color="#e5e5e5", lw=0.9, zorder=0)
    ax.axvline(0, color="#e5e5e5", lw=0.9, zorder=0)

    cam_o = np.array(CAM_POS[:2])
    end_o = np.array(END_POS[:2])

    # 三条边：每条边都是一个 T，箭头方向就是变换方向。
    #   why 前两条用弧线：base 的 +x 轴正好和 base->cam 同向，cam 的 +x 轴正好和
    #       cam->obj 同向（目标就在光轴上），直线画上去会和坐标轴完全重合，
    #       读者分不清哪根是坐标轴、哪根是变换。

    ax.annotate("", xy=cam_o, xytext=np.array([0.0, 0.0]),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2.6,
                                shrinkA=4, shrinkB=8,
                                connectionstyle="arc3,rad=-0.30"))
    ax.text(0.68, -0.32, "T_base_cam", color="#27ae60", fontsize=11,
            fontweight="bold", ha="center")
    ax.annotate("", xy=p_base, xytext=cam_o,
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2.0,
                                linestyle="--", shrinkA=10, shrinkB=10,
                                connectionstyle="arc3,rad=-0.28"))
    ax.text(1.30, 0.80, "T_cam_obj", color="#27ae60", fontsize=11,
            fontweight="bold", ha="left")
    ax.annotate("", xy=end_o, xytext=np.array([0.0, 0.0]),
                arrowprops=dict(arrowstyle="-|>", color="#2c6fbb", lw=2.2,
                                shrinkA=4, shrinkB=8))
    ax.text(0.22, 0.30, "T_base_end", color="#2c6fbb", fontsize=11,
            fontweight="bold", ha="center", rotation=45)
    ax.annotate("", xy=p_base, xytext=end_o,
                arrowprops=dict(arrowstyle="-|>", color="#2c6fbb", lw=1.8,
                                linestyle="--", shrinkA=12, shrinkB=10))
    ax.text(0.60, 0.92, "T_end_obj", color="#2c6fbb", fontsize=11,
            fontweight="bold", ha="center")

    draw_frame(ax, np.array([0.0, 0.0]), 0, "#1f2328", "base", scale=0.26)
    draw_frame(ax, end_o, 0, "#2c6fbb", "end", scale=0.26)
    draw_frame(ax, cam_o, 90, "#27ae60", "cam", scale=0.26, lab_off=(0.11, 0.09))

    ax.plot(*p_base, marker="*", ms=26, color="#c0392b", mec="white", mew=1.2,
            zorder=6)
    ax.text(p_base[0] + 0.06, p_base[1] + 0.10, "目标", color="#c0392b",
            fontsize=12.5, fontweight="bold")

    rows = [
        ("base 系", f"({p_base[0]:.2f}, {p_base[1]:.2f})", "#1f2328"),
        ("end  系", f"({p_end[0]:.2f}, {p_end[1]:.2f})", "#2c6fbb"),
        ("cam  系", f"({p_cam[0]:.2f}, {p_cam[1]:.2f})", "#27ae60"),
    ]
    for i, (name, coord, color) in enumerate(rows):
        ax.text(1.30, 1.72 - i * 0.16, f"{name}：{coord}", color=color,
                fontsize=11.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=color, alpha=0.92))

    ax.set_xlabel("x（米）")
    ax.set_ylabel("y（米）")
    ax.set_title("P4 算例：同一次乘法链，三种读法\n"
                 "绿色两段连乘得到目标在基座系的位置；蓝色一段把目标换到末端系")

    path = OUT / "W3_P4_1_chain.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"写出 {path}")
    print(f"目标在 base 系 ({p_base[0]:.2f}, {p_base[1]:.2f})")
    print(f"目标在 cam  系 ({p_cam[0]:.2f}, {p_cam[1]:.2f})")
    print(f"目标在 end  系 ({p_end[0]:.2f}, {p_end[1]:.2f})")


if __name__ == "__main__":
    main()
