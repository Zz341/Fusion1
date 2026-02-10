import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from models.dwt import DWT_Fixed


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIM(nn.Module):
    def __init__(self, window_size=11, channel=1):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, channel)

    def forward(self, img1, img2):
        if img1.is_cuda:
            self.window = self.window.cuda(img1.get_device())
        self.window = self.window.type_as(img1)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        sigma1_sq = sigma1_sq.clamp(min=0)
        sigma2_sq = sigma2_sq.clamp(min=0)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / denominator
        return ssim_map.mean()


class WaveletLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(WaveletLoss, self).__init__()
        self.dwt = DWT_Fixed().to(device)

    def forward(self, f, a, b):
        f_ll, (f_lh, f_hl, f_hh) = self.dwt(f)
        a_ll, (a_lh, a_hl, a_hh) = self.dwt(a)
        b_ll, (b_lh, b_hl, b_hh) = self.dwt(b)

        # 低频：Max
        loss_ll = F.l1_loss(f_ll, torch.max(a_ll, b_ll))

        # 高频：Selection (保留原始符号)
        loss_high = 0.0
        for (fa, fb, ff) in zip([a_lh, a_hl, a_hh], [b_lh, b_hl, b_hh], [f_lh, f_hl, f_hh]):
            mask = torch.abs(fa) > torch.abs(fb)
            target = fa * mask.float() + fb * (~mask).float()
            loss_high += F.l1_loss(ff, target)

        return loss_ll + loss_high


class FusionLoss(nn.Module):
    def __init__(self, device='cpu', int_w=1.0, grad_w=20.0, ssim_w=3.0, freq_w=5.0):
        super(FusionLoss, self).__init__()
        self.weights = [int_w, grad_w, ssim_w, freq_w]

        # Sobel Kernel
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3).to(device)

        from utils.loss import SSIM  # 假设 SSIM 在同一文件或正确导入
        self.ssim_module = SSIM(window_size=11, channel=1).to(device)
        self.wavelet_module = WaveletLoss(device).to(device)

    def gradient(self, x):
        return F.conv2d(x, self.sobel_x, padding=1), F.conv2d(x, self.sobel_y, padding=1)

    def grad_loss(self, f, a, b):
        gxf, gyf = self.gradient(f)
        gxa, gya = self.gradient(a)
        gxb, gyb = self.gradient(b)

        # 计算梯度幅值
        mag_f = torch.sqrt(gxf ** 2 + gyf ** 2 + 1e-6)
        mag_a = torch.sqrt(gxa ** 2 + gya ** 2 + 1e-6)
        mag_b = torch.sqrt(gxb ** 2 + gyb ** 2 + 1e-6)

        # 目标：每个像素点的梯度幅值应接近 A 和 B 中较大的那个
        target_mag = torch.max(mag_a, mag_b)

        # 原始的 Grad Loss (L1 on directional gradients)
        # mask = mag_a > mag_b
        # target_gx = gxa * mask.float() + gxb * (~mask).float()
        # target_gy = gya * mask.float() + gyb * (~mask).float()
        # l_dir = F.l1_loss(gxf, target_gx) + F.l1_loss(gyf, target_gy)

        # 【新增】幅值 Loss (直接优化 Qabf 的核心)
        l_mag = F.l1_loss(mag_f, target_mag)

        return l_mag * 2.0  # 强化幅值约束

    def forward(self, fused, img_a, img_b):
        l_int = F.l1_loss(fused, torch.max(img_a, img_b))
        l_grad = self.grad_loss(fused, img_a, img_b)
        l_ssim = (1 - self.ssim_module(fused, img_a)) + (1 - self.ssim_module(fused, img_b))
        l_freq = self.wavelet_module(fused, img_a, img_b)

        return (self.weights[0] * l_int +
                self.weights[1] * l_grad +
                self.weights[2] * l_ssim +
                self.weights[3] * l_freq)