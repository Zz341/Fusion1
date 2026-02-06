import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    """
    生成高斯核
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    创建二维高斯窗口
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIM(nn.Module):
    """
    原生 PyTorch 实现的 SSIM 模块 (无需安装第三方库)
    """

    def __init__(self, window_size=11, channel=1):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, channel)

    def forward(self, img1, img2):
        # 如果图片在 GPU 上，高斯核也要搬到 GPU
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

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()


class FusionLoss(nn.Module):
    # 1. 修改 __init__，增加接收权重的参数 (int_w, grad_w, ssim_w)
    def __init__(self, device='cpu', int_w=1.0, grad_w=20.0, ssim_w=5.0):
        super(FusionLoss, self).__init__()
        self.int_w = int_w
        self.grad_w = grad_w
        self.ssim_w = ssim_w
        # 1. Sobel 算子 (用于梯度损失)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(device)

        # 2. SSIM 模块 (用于结构相似性损失)
        # 灰度图 channel=1，窗口大小通常取 11
        self.ssim_module = SSIM(window_size=11, channel=1).to(device)

    def gradient(self, x):
        # 计算梯度
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def l1_loss(self, f, a, b):
        """
        强度损失 (Intensity Loss)
        目标：融合图像的亮度应该接近源图像中较亮的那个 (Max策略) 或者是两者的平均
        对于医学图像，通常希望保留显著特征，这里使用 Max 策略
        """
        target = torch.max(a, b)
        return F.l1_loss(f, target)

    def grad_loss(self, f, a, b):
        """
        梯度损失 (Gradient Loss)
        目标：融合图像的纹理细节(梯度)应该接近源图像中梯度较大的部分
        """
        g_f = self.gradient(f)
        g_a = self.gradient(a)
        g_b = self.gradient(b)
        target_g = torch.max(g_a, g_b)
        return F.l1_loss(g_f, target_g)

    def ssim_loss(self, f, a, b):
        """
        SSIM 损失
        目标：最大化 fused 与 a 的 SSIM，以及 fused 与 b 的 SSIM
        Loss = (1 - SSIM(f, a)) + (1 - SSIM(f, b))
        """
        score_a = self.ssim_module(f, a)
        score_b = self.ssim_module(f, b)

        # SSIM 越接近 1 越好，所以 Loss 是 1 - SSIM
        # 这里给两者各 0.5 的权重，或者直接相加
        return (1 - score_a) + (1 - score_b)

    def forward(self, fused, img_a, img_b):
        # 1. 强度损失 (Pixel-wise)
        l_int = self.l1_loss(fused, img_a, img_b)

        # 2. 梯度损失 (Texture-wise)
        l_grad = self.grad_loss(fused, img_a, img_b)

        # 3. SSIM 损失 (Structure-wise) [现在是真正的 SSIM 了]
        l_ssim = self.ssim_loss(fused, img_a, img_b)

        # 总损失 (权重建议：SSIM 对视觉效果影响最大，可以适当调高)
        # 参考 config.py 中的权重: int=1.0, grad=2.0, ssim=1.0
        # 注意：因为现在的 SSIM loss 实际上是 (1-ssim_a) + (1-ssim_b)，值域在 0~2 之间
        # 之前的 MSE 值域很小。所以 SSIM 的权重可能需要根据训练情况微调。

        # 建议权重：
        # Intensity: 1.0 (基础)
        # Gradient: 5.0 (强调纹理，因为梯度值通常很小)
        # SSIM: 0.5 ~ 1.0 (强调结构)

        # 这里沿用你 config 中的定义，但建议稍微加大 Gradient 的权重
        total_loss = self.int_w * l_int + self.grad_w * l_grad + self.ssim_w * l_ssim

        return total_loss