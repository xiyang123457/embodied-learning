"""画 A.2 要求的那张图：next_error_m 随 step 变化，四条线同轴，纵轴对数。

用途
    让"都在抖"和"都在收敛"在视觉上分不开这件事变成可验证的：DT=0.6 和 DT=1.0 的曲线
    形状相似，但一条到 1e-10 一条停在 0.4 上下——光看图判断不出来，必须算 Kp*DT。
输入
    各配置目录下的 feedback_trace.csv（seed=0 那一回合，100 步）。
输出
    outputs/figures/W2_1_error_vs_step.png
"""

from pathlib import Path
# pathlib：路径处理。

import csv
# csv：读 trace。step 和 next_error_m 读出来是字符串，要转 int / float。

import matplotlib
matplotlib.use("Agg")
# use("Agg")：无窗口后端，只出文件。不设的话某些环境会尝试开窗然后卡住。

import matplotlib.pyplot as plt
# pyplot：画图。标签一律英文——Windows 默认字体没有中文，会变成方框。

TOL = 0.02
# TOL：float，到达容差（米），与 first_loop.py 一致，画成一条参考线。

BASE = Path("outputs/first_loop")
# BASE：Path，各配置的输出目录根。

CURVES = [
    # (trace 路径, 图例文字, 颜色)
    # 顺序即图例顺序：基准在前，三个修改在后。
    (BASE / "feedback_trace.csv", "Kp=2.0 baseline (Kp*DT=0.10)", "tab:blue"),
    (BASE / "kp0.5" / "feedback_trace.csv", "Kp=0.5 slow (Kp*DT=0.025)", "tab:orange"),
    (BASE / "dt0.6" / "feedback_trace.csv", "DT=0.6 overshoot, converges (Kp*DT=1.2)", "tab:green"),
    (BASE / "dt1" / "feedback_trace.csv", "DT=1.0 limit cycle, never settles (Kp*DT=2.0)", "tab:red"),
]


def load(path):
    """读一个 trace 文件。返回 (steps, errors)，都是 list。"""
    steps, errors = [], []
    with path.open(encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            steps.append(int(rec["step"]))
            errors.append(float(rec["next_error_m"]))
            # float()：CSV 读出来是字符串，不转换后面画不了图。
    return steps, errors


def main():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    # figsize：英寸。加宽到 9.5 是为了把图例放到画布外，不压住 DT=1.0 那条水平线。

    for path, label, color in CURVES:
        steps, errors = load(path)
        # 误差精确等于 0 时对数轴无定义（log(0) = -inf）。用 nan 让线在这里自然截断，
        # 而不是垫一个 1e-18 的假值——那会在图底画出一条不存在的水平线。
        ax.semilogy(steps, [e if e > 0 else float("nan") for e in errors],
                    label=label, color=color, lw=1.6)

    ax.axhline(TOL, color="gray", ls="--", lw=1.0)
    # axhline：容差参考线。线在它下面 = 进入了容差，比用文字描述直观。
    # 标签放在 (68, 0.0085)：这一小块是唯一没有曲线经过的位置
    #   （蓝线在 x=68 已是 1e-4 量级，橙线 0.045，红线 0.4），避免字被线穿过。
    ax.text(68, TOL * 0.42, "tolerance = 0.02 m", color="gray", fontsize=8)

    ax.set_xlabel("step (DT = 0.05 s each, 100 steps = 5 s)")
    ax.set_ylabel("next_error_m [m]  (log scale)")
    ax.set_title("Appendix A: position error vs step (seed 0)")
    ax.set_ylim(1e-16, 1.2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    # bbox_to_anchor 放到画布右侧外：四条线的图例字太长，放里面必然压线。

    # 用 ax.text 而不是 annotate：annotate 的箭头会在图上画出一条直线，
    #   读者会把它当成第五条曲线。这里把说明放在图底空白处。
    ax.text(24, 2.5e-16, "DT=0.6 reaches exact 0.0 at step 20 (float precision); curve ends there",
            fontsize=7, color="tab:green", va="bottom")

    out = Path("outputs/figures/W2_1_error_vs_step.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    # dpi=150：在 7.5x4.5 英寸上相当于 1125x675，够看清对数轴上的抖动。
    print(f"Saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
