import torch
import torch.nn as nn


class ResNetBackbone(nn.Module):
    def __init__(self, input_channels=1, output_channels=64):
        super(ResNetBackbone, self).__init__()

        # --- 修改部分开始: Deep Stem ---
        # 目的：使用连续的3x3卷积替代7x7大卷积，且 stride=1 保持原分辨率
        self.stem = nn.Sequential(
            # 第一层：3x3, stride=1
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 第二层：3x3, stride=1 (增加非线性)
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 第三层：3x3, stride=1 (映射到目标维度)
            nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        # --- 修改部分结束 ---

        # 这里的 output_channels 默认是 64
        # 输出特征图尺寸: [B, 64, H, W] (原尺寸)

    def forward(self, x):
        x = self.stem(x)
        return x