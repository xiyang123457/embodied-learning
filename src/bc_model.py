"""W6 · BC 模型：2 层隐藏层 × 128，ReLU，输入 8 维、输出 3 维。

参数量：8×128+128 + 128×128+128 + 128×3+3 = 18,051
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.bc_dataset import obs_to_input

INPUT_DIM = 8
HIDDEN = 128
OUTPUT_DIM = 3


class BCPolicy(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = HIDDEN, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TorchPolicy:
    """闭环测试用的推理包装：输入原始 8 维（未归一化），输出 4 维动作（clip 到 [-1,1]）。"""

    def __init__(self, model: BCPolicy, x_mean, x_std):
        self.model = model.eval()
        self.x_mean = torch.as_tensor(x_mean, dtype=torch.float32)
        self.x_std = torch.as_tensor(x_std, dtype=torch.float32)

    def __call__(self, obs, info=None):
        x = obs_to_input(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
        with torch.no_grad():
            xb = (torch.as_tensor(x, dtype=torch.float32) - self.x_mean) / self.x_std
            a = self.model(xb).numpy()[0]
        out = np.zeros(4, dtype=np.float32)
        out[:3] = np.clip(a, -1.0, 1.0)
        return out
