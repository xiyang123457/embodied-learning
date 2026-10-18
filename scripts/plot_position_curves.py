"""画"位置相对目标的偏移"随 step 变化，用来看见误差图上被绝对值吞掉的过冲。

用途
    误差图（W2_1）画的是 |e|，而 |e| 在 DT=0.6 时是单调下降的——过冲在图上完全消失。
    要看"来回跳"，必须画**带符号**的 px - gx：DT=0.6 每步反向但幅度 ×0.2，
    DT=1.0 每步反向且幅度不变（极限环）。同一张图上才分得清这四种情形。
输入
    outputs/first_loop/ 下各配置的 feedback_trace.csv（seed=0）。
输出
    outputs/figures/W2_2_position_oscillation.png
"""

from pathlib import Path
# pathlib：路径处理。

import csv
# csv：读 trace；px_m / gx_m 读出来是字符串，要 float()。

import matplotlib
matplotlib.use("Agg")
# Agg：无窗口后端，只出文件。

import matplotlib.pyplot as plt
# pyplot：画图；标签用英文，避免 Windows 中文字体变方框。

BASE = Path("outputs/first_loop")

CURVES = [
    # (trace 路径, 图例, 颜色)
    # 只画 x 方向一条轴：另一轴行为完全相同，画两条没有额外信息。
    (BASE / "feedback_trace.csv", "Kp=2.0 baseline (Kp*DT=0.10, no overshoot)", "tab:blue"),
    (BASE / "dt0.6" / "feedback_trace.csv", "DT=0.6 (Kp*DT=1.2): flips sign, amplitude x0.2", "tab:green"),
    (BASE / "dt1" / "feedback_trace.csv", "DT=1.0 (Kp*DT=2.0): flips sign, amplitude constant for all 100 steps", "tab:red"),
]


def load_offset(path):
    """返回 (steps, px - gx)。px - gx 的单位是米，带符号，0 = 已对齐目标。"""
    steps, offsets = [], []
    with path.open(encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            steps.append(int(rec["step"]))
            offsets.append(float(rec["px_m"]) - float(rec["gx_m"]))
            # 用"相对目标的偏移"而不是绝对坐标：三条线的目标本来相同，
            #   减掉之后 0 就是目标，正负号直接表示在目标哪一侧。
    return steps, offsets


def main():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))

    for path, label, color in CURVES:
        steps, offsets = load_offset(path)
        ax.plot(steps, offsets, label=label, color=color, lw=1.6)

    ax.axhline(0.0, color="black", lw=0.8)
    # 0 = 目标所在的一侧；线穿过它就是"冲过目标"。

    ax.set_yscale("symlog", linthresh=1e-4)
    # symlog：0 附近用线性、远处用对数。普通 log 画不出负值和 0，
    #   而 DT=0.6 的后面几十步幅度是 1e-9 量级，纯线性会把它们压成一条线。

    ax.set_xlabel("step (DT = 0.05 s each)  —  first 20 steps shown")
    ax.set_ylabel("px - gx [m]  (signed offset, symlog)")
    ax.set_title("Signed offset from goal (seed 0): overshoot is hidden by |e|", loc="left")
    ax.set_xlim(-0.5, 20.5)
    # 只画前 20 步：DT=0.6 的振幅 3 步内就乘到 1e-3 以下，在 100 步的横轴上会被压成
    #   纵轴边的一条线。DT=1.0 的"幅度不衰减"由 W2_1 里那条 100 步的水平线证明。
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    out = Path("outputs/figures/W2_2_position_oscillation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
