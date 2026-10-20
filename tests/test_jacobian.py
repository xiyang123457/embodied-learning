"""src/jacobian.py 的单测。

用途
    三条防线，缺一不可：
    1. 已知算例（(0,0) 代入得 [[0,0],[2,1]]）——  唯一能抓「行列写反」和「漏项」的
    2. 解析 vs 中心差分  ——  能抓「解析式和自己写的偏导不一致」
    3. det = l1·l2·sinθ2  ——  把奇异条件压成一行，顺便验证 l1、l2 没被写死成 1
    第 1 条不能省：第 2 条只证明「两个实现自洽」，两个一起错时它照样全绿。

输入
    无，全部用固定位形，任何机器上结果一致。

输出
    python -m pytest tests/test_jacobian.py -q 打印 6 passed。

跑完应看到的自检数字
    6 passed
    (0, 0) 处 J = [[0, 0], [2, 1]]
    三个位形的相对误差都 < 1e-8
"""

import numpy as np
import pytest

from src.jacobian import J_ana, J_fd, cond_num, det_ana
from src.two_link import fk

ATOL = 1e-8
# ATOL：float，相对误差阈值。
#   为什么定 1e-8 而不是 1e-12：中心差分在 ε=1e-6 时理论误差是 O(ε²)=1e-12，
#   但浮点本身还有 ~1e-16 的底噪，实际落在 1e-10 量级。
#   定 1e-8 留两个数量级余量，同时仍能把「用了前向差分」这种错（误差 1e-6）抓出来


# ==== 1. 已知算例：定海神针 ====
# 手臂沿 +x 伸直。只转 θ1 时末端速度 (0,2)，只转 θ2 时 (0,1)，拼起来就是 [[0,0],[2,1]]。
# 这条能抓两类错：行列写反、第二列漏成和第一列一样。

def test_known_configuration():
    """θ=(0,0) 时 J 必须是 [[0,0],[2,1]]，x 方向速度为 0。"""
    J = J_ana((0.0, 0.0))
    assert np.allclose(J, [[0.0, 0.0], [2.0, 1.0]], atol=1e-12)
    assert np.allclose(J[0], 0.0), "第一行（x 方向速度）在伸直位形下必须是 0"


# ==== 2. 解析 vs 中心差分 ====
# 三个位形覆盖：奇异（伸直）、W3 那个 (0,√2) 姿态、一般位形。

@pytest.mark.parametrize("theta", [(0.0, 0.0), (np.pi / 4, np.pi / 2), (0.3, 1.2)])
def test_analytic_matches_finite_difference(theta):
    """解析解与中心差分的相对误差应小于 1e-8。"""
    Ja = J_ana(theta)
    Jf = J_fd(theta)
    rel = np.abs(Ja - Jf).max() / (np.abs(Ja).max() + 1e-12)
    assert rel < ATOL, f"相对误差 {rel:.2e}，检查差分公式或 eps"


def test_forward_difference_is_worse():
    """前向差分的误差应该明显大于中心差分，用来确认 eps 与公式都选对了。

    这条不是在测代码，是在测「测试本身有没有鉴别力」。
    如果前向和中心算出来的误差一样大，说明这个位形根本没被测到敏感方向。
    """
    theta = (0.3, 1.2)
    Ja = J_ana(theta)

    eps = 1e-6
    f0 = np.array(fk(*theta))
    J_forward = np.column_stack([
        (np.array(fk(theta[0] + eps, theta[1])) - f0) / eps,
        (np.array(fk(theta[0], theta[1] + eps)) - f0) / eps,
    ])

    err_forward = np.abs(Ja - J_forward).max()
    err_central = np.abs(Ja - J_fd(theta)).max()
    assert err_central < err_forward / 100.0


# ==== 3. 行列式 ====
# det 有两个作用：验证 l1、l2 没被写死，以及给出奇异条件。

@pytest.mark.parametrize("t2", [0.0, 0.5, 1.2, np.pi])
def test_det_matches_closed_form(t2):
    """np.linalg.det(J_ana) 应等于 l1·l2·sin(θ2)。"""
    theta = (0.7, t2)
    assert np.allclose(np.linalg.det(J_ana(theta)), 1.0 * 1.0 * np.sin(t2), atol=1e-12)


def test_det_uses_link_lengths():
    """杆长不是 1 时 det 必须跟着变，否则说明公式里 l1、l2 被写死了。"""
    J = J_ana((0.4, 0.9), l1=2.0, l2=0.5)
    assert np.allclose(np.linalg.det(J), det_ana(0.9, l1=2.0, l2=0.5), atol=1e-12)
    assert np.allclose(det_ana(0.9, l1=2.0, l2=0.5), 2.0 * 0.5 * np.sin(0.9), atol=1e-12)


def test_singular_at_full_extension_and_fold():
    """θ2 = 0（伸直）与 θ2 = π（折叠）都是奇异位形，条件数为 inf。"""
    assert np.allclose(det_ana(0.0), 0.0, atol=1e-12)
    assert np.isinf(cond_num((0.0, 0.0))) or cond_num((0.0, 0.0)) > 1e15
    assert np.allclose(det_ana(np.pi), 0.0, atol=1e-12)
