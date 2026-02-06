import torch
import torch.nn as nn


class DWT_Fixed(nn.Module):
    def __init__(self):
        super(DWT_Fixed, self).__init__()
        # 定义Haar小波的四个滤波器 (LL, LH, HL, HH)
        # 1/2 是归一化系数，保证能量守恒或数值稳定
        self.requires_grad = False

    def forward(self, x):
        # x: [B, C, H, W]
        # 简单的切片操作实现Haar变换，比卷积更快且无参数
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        # LL: 低频 (平均)
        x_LL = x1 + x2 + x3 + x4
        # LH: 水平细节
        x_LH = -x1 - x2 + x3 + x4
        # HL: 垂直细节
        x_HL = -x1 + x2 - x3 + x4
        # HH: 对角细节
        x_HH = x1 - x2 - x3 + x4

        return x_LL, (x_LH, x_HL, x_HH)


class IDWT_Fixed(nn.Module):
    def __init__(self):
        super(IDWT_Fixed, self).__init__()
        self.requires_grad = False

    def forward(self, x_LL, x_high):
        x_LH, x_HL, x_HH = x_high

        # 逆变换公式 (保持不变)
        r1 = x_LL - x_LH - x_HL + x_HH
        r2 = x_LL - x_LH + x_HL - x_HH
        r3 = x_LL + x_LH - x_HL - x_HH
        r4 = x_LL + x_LH + x_HL + x_HH

        # [关键修改在这里]
        B, C, H, W = x_LL.shape

        # 原代码: x_rec = torch.zeros(B, C, H * 2, W * 2).to(x_LL.device)
        # 问题: 这会先占用CPU内存，再移动到GPU。

        # 修改后: 直接在 device 上创建，不占用 CPU 内存
        x_rec = torch.zeros(B, C, H * 2, W * 2, device=x_LL.device)

        # 赋值操作 (保持不变)
        x_rec[:, :, 0::2, 0::2] = r1
        x_rec[:, :, 1::2, 0::2] = r2
        x_rec[:, :, 0::2, 1::2] = r3
        x_rec[:, :, 1::2, 1::2] = r4

        return x_rec