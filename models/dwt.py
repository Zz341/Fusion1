import torch
import torch.nn as nn

class DWT_Fixed(nn.Module):
    def __init__(self):
        super(DWT_Fixed, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # 1. 切片 (保持原始数值，不除以2)
        x01 = x[:, :, 0::2, :]
        x02 = x[:, :, 1::2, :]
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        # 2. 低频 = 平均值 (保持 0-1 分布)
        x_LL = (x1 + x2 + x3 + x4) / 4

        # 3. 高频 = 差分/2 (防止数值过大)
        x_LH = (-x1 - x2 + x3 + x4) / 2
        x_HL = (-x1 + x2 - x3 + x4) / 2
        x_HH = (x1 - x2 - x3 + x4) / 2

        return x_LL, (x_LH, x_HL, x_HH)


class IDWT_Fixed(nn.Module):
    def __init__(self):
        super(IDWT_Fixed, self).__init__()
        self.requires_grad = False

    def forward(self, x_LL, x_high):
        x_LH, x_HL, x_HH = x_high

        # 4. 完美重构公式
        # x1 = Avg - Diff_Avg
        x1 = x_LL - (x_LH + x_HL - x_HH) / 2
        x2 = x_LL - (x_LH - x_HL + x_HH) / 2
        x3 = x_LL + (x_LH - x_HL - x_HH) / 2
        x4 = x_LL + (x_LH + x_HL + x_HH) / 2

        B, C, H, W = x_LL.shape
        x_out = torch.zeros(B, C, H * 2, W * 2).type_as(x_LL)

        x_out[:, :, 0::2, 0::2] = x1
        x_out[:, :, 1::2, 0::2] = x2
        x_out[:, :, 0::2, 1::2] = x3
        x_out[:, :, 1::2, 1::2] = x4

        return x_out