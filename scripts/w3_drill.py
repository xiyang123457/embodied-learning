"""W3 闭卷自检：把你写的 inverse / apply 和标准实现逐项对比。

用途
    只测 src/transforms_drill.py 里的两个函数。
    不测 rot_x/y/z 和 make_T，因为它们没有判断点；
    也不测 compose，因为它和 inverse/apply 不同源，写错的概率低。

    测法不是「跑一个例子看数字」，而是拿随机生成的一批合法变换和点集逐个比：
    单个例子可能恰好落在两种实现重合的地方（比如平移为 0 时 -t 和 -Rᵀt 相等），
    随机批量的目的是让这种巧合失效。

输入
    src/transforms_drill.py 里填好的 inverse 和 apply。

输出
    控制台打印两个函数的 PASS / FAIL，最后一行给总体结论。

跑完应看到的自检数字
    inverse 最大偏差 0.0
    apply   最大偏差 0.0
    全绿: True
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transforms import apply as ref_apply
from src.transforms import inverse as ref_inverse
from src.transforms import make_T, rot_x, rot_y, rot_z

try:
    from src.transforms_drill import apply as my_apply
    from src.transforms_drill import inverse as my_inverse
except (ImportError, NotImplementedError) as exc:
    print(f"还没开始：{exc}")
    print("先填 src/transforms_drill.py 里那两个函数体。")
    sys.exit(2)


rng = np.random.default_rng(0)
# rng：Generator，固定种子。目的是让「上次没通过、这次通过」可比，
#   换种子后如果通过数上升，说明是随机撞对的，很可能是错的


def random_T():
    """生成一个随机但合法的 4x4 齐次变换（随机旋转 + 随机平移）。"""
    R = rot_z(rng.uniform(-np.pi, np.pi)) @ rot_x(rng.uniform(-np.pi, np.pi)) \
        @ rot_y(rng.uniform(-np.pi, np.pi))
    # 三个基本旋转连乘，结果必然是正交阵且 det = +1，不会出现镜像
    return make_T(R, rng.uniform(-2.0, 2.0, size=3))


def check_inverse(n=50):
    """对比两个 inverse，返回 (最大偏差, 往返误差)。"""
    worst_inv, worst_round = 0.0, 0.0
    for _ in range(n):
        T = random_T()
        mine = np.asarray(my_inverse(T), dtype=float)
        worst_inv = max(worst_inv, float(np.abs(mine - ref_inverse(T)).max()))
        worst_round = max(worst_round, float(np.abs(mine @ T - np.eye(4)).max()))
    return worst_inv, worst_round


def check_apply(n=50):
    """对比两个 apply，覆盖单点 (3,) 与批量 (N,3) 两种输入。"""
    worst, shape_ok = 0.0, True
    for _ in range(n):
        T = random_T()
        pts = rng.uniform(-3.0, 3.0, size=(rng.integers(1, 8), 3))
        out = np.asarray(my_apply(T, pts), dtype=float)
        shape_ok = shape_ok and out.shape == pts.shape
        worst = max(worst, float(np.abs(out - ref_apply(T, pts)).max()))

        single = np.asarray(my_apply(T, pts[0]), dtype=float)
        shape_ok = shape_ok and single.shape == (1, 3)
        worst = max(worst, float(np.abs(single - ref_apply(T, pts[0])).max()))
    return worst, shape_ok


def main():
    # Windows 控制台默认 GBK，打印上标、箭头这类字符会抛 UnicodeEncodeError，
    #   把 stdout 固定成 utf-8，脚本才不会在自己打印报告的时候崩掉
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        worst_inv, worst_round = check_inverse()
        inv_ok = worst_inv < 1e-12 and worst_round < 1e-12
    except NotImplementedError:
        print("  inverse 还是 raise NotImplementedError")
        worst_inv, inv_ok = float("nan"), False

    try:
        worst_apply, shape_ok = check_apply()
        ap_ok = worst_apply < 1e-12 and shape_ok
    except NotImplementedError:
        print("  apply 还是 raise NotImplementedError")
        worst_apply, ap_ok = float("nan"), False

    print("== 闭卷自检 ==")
    print(f"  inverse 与标准实现最大偏差 {worst_inv:.6g}")
    print(f"  apply   与标准实现最大偏差 {worst_apply:.6g}")
    print()
    print(("  PASS  " if inv_ok else "  FAIL  ") + "inverse（含 T 乘它的逆 = 单位阵）")
    print(("  PASS  " if ap_ok else "  FAIL  ") + "apply（含 (3,) 与 (N,3) 两种形状）")
    print("\n全绿:", inv_ok and ap_ok)

    if not inv_ok:
        print("  inverse 常见错法：平移写成 -t（只在 R 为单位阵时才对）。")
    if not ap_ok:
        print("  apply 常见错法：写成 Ph @ T（旋转反向、平移丢失），或忘了外层转置。")


if __name__ == "__main__":
    main()
