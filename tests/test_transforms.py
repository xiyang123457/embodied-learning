"""src/transforms.py 的六条单测。

用途
    给坐标变换小库买保险。W4 的雅可比有限差分、W5 的轨迹坐标转换、W6 的 BC
    输入预处理全部建立在 transforms.py 正确之上，所以这六条不是形式，是上游依赖。
    每条测试的 docstring 都写了「它能抓到什么错」，写新测试时照这个格式补。

输入
    无。全部用固定的手算值，不依赖随机数，任何机器上结果一致。

输出
    `python -m pytest tests/ -q` 打印 6 passed。

跑完应看到的自检数字
    6 passed，总耗时 < 1 秒
    把 rot_y 里 -s 挪到右上角后，只有 test_known_answer 会失败，其余五条照样绿
"""

import numpy as np
import pytest
# pytest：测试框架。作用是把每条 test_ 函数独立执行、独立报错、独立统计通过数。
#   关键参数：-q 只打一行摘要，-v 打印每条测试名
#   坑：文件名必须以 test_ 开头、函数名必须以 test_ 开头，否则 pytest 静默跳过，
#       你会看到 "no tests ran" 而不是报错

from src.transforms import apply, compose, inverse, make_T, rot_x, rot_y, rot_z

ATOL = 1e-12
# ATOL：float，绝对容差
#   为什么是 1e-12：绕 90 度转四次的累积误差在 1e-16 量级，留四位余量足够；
#   若放宽到 1e-6，inverse 里把 -Rᵀt 写成 -t 这种"差 1 米"的错依然能抓到，
#   但把 R 的某个 1e-3 量级符号错漏掉。松紧两头都要考虑


# ==== 1. 转四次等于没转 ====
# 这条抓"角度单位搞错"。把 pi/2 写成 90 时，训练轮 loop 依然跑得通，
# 但 cos(90) = -0.448，矩阵完全变形，四次相乘离单位阵很远。

def test_four_quarter_turns_are_identity():
    """R_z(90°) 连乘四次应回到单位阵，抓角度单位写成角度制。"""
    R = rot_z(np.pi / 2)
    assert np.allclose(R @ R @ R @ R, np.eye(3), atol=ATOL)


# ==== 2. 正交性与 det ====
# 直接对三个轴都测，参数化后一条函数覆盖三种情况。
# 这条抓"矩阵元素写错"和"写成了镜像"：翻成镜像时 RᵀR 仍是单位阵，
# 但 det 变成 -1，所以两个断言必须同时存在，只测 RᵀR 抓不到镜像。

@pytest.mark.parametrize("R", [rot_x(0.7), rot_y(-1.2), rot_z(2.5)])
def test_rotation_is_orthonormal_and_proper(R):
    """任取一个角度，RᵀR 应为单位阵，det 应为 +1（-1 是镜像不是旋转）。"""
    assert np.allclose(R.T @ R, np.eye(3), atol=ATOL)
    assert np.allclose(np.linalg.det(R), 1.0, atol=ATOL)


# ==== 3. 逆变换 ====
# 这条专门抓 inverse 里 -Rᵀt 写成 -t。取 t=(1,0,0) 是为了让两种写法结果明显不同：
# 正确是 (0,1,0)，写错是 (-1,0,0)。若取 t=(0,0,0)，两种写法都对，这条就白写了。

def test_inverse_is_true_inverse():
    """T @ T⁻¹ 与 T⁻¹ @ T 都应等于单位阵，抓 -Rᵀt 写成 -t。"""
    T = make_T(rot_z(np.pi / 2), [1.0, 0.0, 0.0])
    assert np.allclose(T @ inverse(T), np.eye(4), atol=ATOL)
    assert np.allclose(inverse(T) @ T, np.eye(4), atol=ATOL)


# ==== 4. 点往返 ====
# 端到端检查：不只看矩阵乘出来像不像单位阵，而是看一个真实点能不能回到原位。
# 顺带验证 apply 的批量路径：传 (3,3) 进来必须出 (3,3)，不能塌成单点。

def test_points_round_trip():
    """三个点变过去再变回来，误差应小于 1e-12，同时验证批量形状。"""
    T = make_T(rot_z(np.pi / 3), [0.3, -0.7, 1.1])
    pts = np.array([[1.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0],
                    [-1.0, 0.5, 2.0]])

    back = apply(inverse(T), apply(T, pts))
    assert back.shape == (3, 3)
    assert np.allclose(back, pts, atol=ATOL)


# ==== 5. 复合顺序 ====
# 这是本周的核心测试。先算合法的那条链，再算反序的那条，两条都必须被检查：
# 只检查"合法链结果对"，无法证明"反序链会错"，也就无法证明这条规则有约束力。

def test_compose_order_matters():
    """T_base_cam @ T_cam_obj 应得 (1,1,0)；反序复合必须给出别的值。"""
    T_base_cam = make_T(rot_z(np.pi / 2), [1.0, 0.0, 0.0])
    T_cam_obj = make_T(np.eye(3), [1.0, 0.0, 0.0])

    T_base_obj = compose(T_base_cam, T_cam_obj)
    assert np.allclose(apply(T_base_obj, [0.0, 0.0, 0.0]), [[1.0, 1.0, 0.0]], atol=ATOL)

    T_reverse = compose(T_cam_obj, T_base_cam)
    assert not np.allclose(apply(T_reverse, [0.0, 0.0, 0.0]), [[1.0, 1.0, 0.0]], atol=ATOL)


# ==== 6. 已知算例 ====
# 最小可验证单位。这条是唯一能抓住 R_y 符号写反的测试：
# 把 -s 挪到右上角得到的矩阵是 R_y(-90°)，它依然是正交阵、det 依然是 +1，
# 所以第 2 条和第 3 条都放它过去。只有"某个已知向量应该转到某个已知方向"能抓。

@pytest.mark.parametrize("axis,expected", [("z", [0.0, 1.0, 0.0]), ("y", [1.0, 0.0, 0.0])])
def test_known_answer(axis, expected):
    """绕 z 转 90° 把 x 轴送到 y；绕 y 转 90° 把 z 轴送到 x。"""
    if axis == "z":
        got = rot_z(np.pi / 2) @ np.array([1.0, 0.0, 0.0])
    else:
        got = rot_y(np.pi / 2) @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(got, expected, atol=ATOL)
