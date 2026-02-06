import torch
import torch.nn.functional as F
import numpy as np
import math


class FusionMetrics:
    def __init__(self, device='cpu'):
        self.device = device

    def _normalize(self, img):
        """将图像归一化到 0-255 范围 (用于计算指标)"""
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        # 假设输入是 [B, C, H, W] 或 [H, W]
        if img.ndim == 4:
            img = img[0, 0, :, :]
        elif img.ndim == 3:
            img = img[0, :, :]

        img = img.astype(np.float32)
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val) * 255.0
        return img

    def EN(self, fused):
        """信息熵 (Entropy)"""
        img = self._normalize(fused).astype(np.uint8)
        hist, _ = np.histogram(img, bins=256, range=(0, 255))
        prob = hist / float(img.size)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))

    def SD(self, fused):
        """标准差 (Standard Deviation)"""
        img = self._normalize(fused)
        return np.std(img)

    def SF(self, fused):
        """空间频率 (Spatial Frequency)"""
        img = self._normalize(fused)
        RF = np.diff(img, axis=0)
        RF = np.sqrt(np.mean(RF ** 2))
        CF = np.diff(img, axis=1)
        CF = np.sqrt(np.mean(CF ** 2))
        return np.sqrt(RF ** 2 + CF ** 2)

    def AG(self, fused):
        """平均梯度 (Average Gradient)"""
        img = self._normalize(fused)
        grad_x = np.diff(img, axis=1)
        grad_y = np.diff(img, axis=0)
        # 对齐尺寸
        w = min(grad_x.shape[1], grad_y.shape[1])
        h = min(grad_x.shape[0], grad_y.shape[0])
        grad_x = grad_x[:h, :w]
        grad_y = grad_y[:h, :w]
        return np.mean(np.sqrt((grad_x ** 2 + grad_y ** 2) / 2))

    def MI(self, fused, img_a, img_b):
        """互信息 (Mutual Information) - MI(F,A) + MI(F,B)"""
        f = self._normalize(fused).astype(np.uint8)
        a = self._normalize(img_a).astype(np.uint8)
        b = self._normalize(img_b).astype(np.uint8)

        def get_mi(im1, im2):
            hist_2d, _, _ = np.histogram2d(im1.flatten(), im2.flatten(), bins=256, range=[[0, 255], [0, 255]])
            pxy = hist_2d / float(np.sum(hist_2d))
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)
            px_py = px[:, None] * py[None, :]
            nzs = pxy > 0
            return np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))

        return get_mi(f, a) + get_mi(f, b)

    def SCD(self, fused, img_a, img_b):
        """差异相关和 (Sum of Correlations of Differences)"""
        f = self._normalize(fused)
        a = self._normalize(img_a)
        b = self._normalize(img_b)

        diff_a = f - a
        diff_b = f - b

        # 计算相关系数
        def correlation(im1, im2):
            mean1, mean2 = np.mean(im1), np.mean(im2)
            num = np.sum((im1 - mean1) * (im2 - mean2))
            den = np.sqrt(np.sum((im1 - mean1) ** 2)) * np.sqrt(np.sum((im2 - mean2) ** 2))
            if den == 0: return 0
            return num / den

        return correlation(diff_a, a) + correlation(diff_b, b)

    def Qabf(self, fused, img_a, img_b):
        """
        Qabf (Edge Preservation Index) - Xydeas & Petrovic (2000)
        这是最复杂但也最受认可的指标。
        """
        f = self._normalize(fused)
        a = self._normalize(img_a)
        b = self._normalize(img_b)

        # Sobel算子参数
        import scipy.signal as signal

        # 定义 Sobel 核
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        def get_gradient(img):
            gx = signal.convolve2d(img, sobel_x, mode='same')
            gy = signal.convolve2d(img, sobel_y, mode='same')
            g_strength = np.sqrt(gx ** 2 + gy ** 2)
            g_alpha = np.arctan2(gy, gx)  # 角度
            return g_strength, g_alpha

        g_a, alpha_a = get_gradient(a)
        g_b, alpha_b = get_gradient(b)
        g_f, alpha_f = get_gradient(f)

        # Qabf 参数 (来源于原始论文)
        L, Tg, kg, Dg, Ta, ka, Da = 1, 0.9994, -15, 0.5, 0.9879, -22, 0.8

        def sigmoid(x, T, k, D):
            return D + (1 - D) / (1 + np.exp(-k * (x - T)))  # 论文公式修正
            # 通常简化实现: Q = Gamma / (1 + exp(k*(x-T)))
            # 这里使用标准 Qabf 实现库中的逻辑

        def edge_preservation(g1, a1, g2, a2):
            # 强度保留 Qg
            # Qg = Tg / (1 + exp(kg * (G1 - G2) / (max(G1, G2) + eps)))
            # 简化版逻辑，参考常用 Matlab 代码转译:

            # 1. 强度因子 Qg
            val_g = np.zeros_like(g1)
            mask = (g1 == 0) & (g2 == 0)
            # 避免除0
            denom = np.maximum(g1, g2)
            # 防止极小值
            denom[denom == 0] = 1e-6

            # 这种实现方式比较复杂，我们使用一种更稳定的标准计算方式
            # 来源: "Objective video quality assessment methods..."

            # 强度相似度
            Qg = Tg / (1 + np.exp(kg * (np.abs(g1 - g2) / denom - Dg)))

            # 角度相似度
            diff_alpha = np.abs(a1 - a2)
            # 归一化到 0-pi
            diff_alpha = np.mod(diff_alpha, 2 * np.pi)
            diff_alpha[diff_alpha > np.pi] = 2 * np.pi - diff_alpha[diff_alpha > np.pi]

            Qa = Ta / (1 + np.exp(ka * (diff_alpha / (np.pi / 2) - Da)))

            return Qg * Qa

        Q_af = edge_preservation(g_a, alpha_a, g_f, alpha_f)
        Q_bf = edge_preservation(g_b, alpha_b, g_f, alpha_f)

        # 权重 (通常使用较大的梯度值作为权重)
        w_a = g_a ** L
        w_b = g_b ** L
        sum_w = w_a + w_b

        # 避免分母为0
        sum_w[sum_w == 0] = 1e-6

        numerator = Q_af * w_a + Q_bf * w_b
        Q = np.sum(numerator) / np.sum(sum_w)
        return Q

    def MS_SSIM(self, fused, img_a, img_b):
        """结构相似性 (仅调用你之前的 pytorch 实现)"""
        # 注意：这里我们简单复用你之前的 loss.py 里的逻辑，或者使用 skimage
        # 为了不依赖库，我们用一个简化版 (A+B)/2 的 SSIM
        from utils.loss import SSIM
        ssim_tool = SSIM(window_size=11, channel=1).to(self.device)

        # 转 Tensor
        def to_tensor(x):
            return torch.from_numpy(self._normalize(x)).unsqueeze(0).unsqueeze(0).to(self.device) / 255.0

        f_t = to_tensor(fused)
        a_t = to_tensor(img_a)
        b_t = to_tensor(img_b)

        score = 0.5 * ssim_tool(f_t, a_t) + 0.5 * ssim_tool(f_t, b_t)
        return score.item()