"""W3 闭卷重写练习：只填 inverse 和 apply 两个函数。

为什么只练这两个
    rot_x/y/z 是照抄矩阵，make_T 是三次赋值，compose 是一个循环，
    它们没有判断点，写一遍也没有增量。
    inverse 和 apply 各有一个「不报错但结果错」的分支：
        inverse：平移项的逆是 -Rᵀt，不是 -t
        apply  ：点要左乘矩阵，写成 Ph @ T 会反向旋转并丢掉平移
    这两个才是要练的。

用法
    1. 先把 src/transforms.py 关掉，不要看同名函数
    2. 把下面两个 raise 换成你的实现
    3. 在仓库根目录跑 `python scripts/w3_drill.py`
    4. 全绿之后这个文件可以删掉，也可以留着当复习卡

跑完应看到的结果
    scripts/w3_drill.py 打印两个函数的 PASS，最后一行「全绿: True」
"""

import numpy as np


def inverse(T):
    """返回 (4,4) 逆矩阵。提示：不要调 np.linalg.inv，用旋转正交这条性质。"""
    R,t = T[:3,:3],T[:3,3]

    Ti=np.eye(4)

    Ti[:3,:3]=R.T
    Ti[:3,3] = -R.T@t

    return Ti



def apply(T, points):
    """把点从 source 系变到 target 系。points 为 (3,) 或 (N,3)，返回 (N,3)。"""
    P=np.atleast_2d(np.asarray(points, dtype=float))

    Ph=np.hstack([P,np.ones((P.shape[0],1))])

    return (T@Ph.T).T[:,:3]
    

