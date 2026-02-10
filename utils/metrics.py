import torch
import torch.nn.functional as F
import numpy as np
import math
import scipy.signal as signal


class FusionMetrics:
    def __init__(self, device='cpu'):
        self.device = device

    def _normalize(self, img):
        """
        将图像从 [0, 1] 线性映射到 [0, 255]
        假设输入 img 已经是归一化好的 tensor 或 numpy 数组
        """
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()

        # 处理维度: [B, C, H, W] -> [H, W]
        if img.ndim == 4:
            img = img[0, 0, :, :]
        elif img.ndim == 3:
            img = img[0, :, :]

        img = img.astype(np.float32)

        # 【关键修改】直接乘以 255，不要减去 min_val
        # 这样保留了图像原本的明暗关系和对比度
        img = img * 255.0

        # 防止精度溢出（虽然理论上不会超过 255）
        img[img > 255] = 255
        img[img < 0] = 0

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

    def VIF(self, fused, img_a, img_b):
        """
        视觉信息保真度 (Visual Information Fidelity, VIF)
        使用 Pixel-domain VIF (VIFp) 实现
        VIF = VIF(Fused, A) + VIF(Fused, B)
        """
        f = self._normalize(fused)
        a = self._normalize(img_a)
        b = self._normalize(img_b)

        def vif_single(ref, dist):
            sigma_nsq = 2.0
            eps = 1e-10

            num = 0.0
            den = 0.0

            # 使用高斯核计算局部统计量
            sigma = 2.0
            # 窗口大小
            win_size = int(6 * sigma + 1)
            if win_size % 2 == 0: win_size += 1

            # 创建高斯核
            k1d = signal.gaussian(win_size, std=sigma).reshape(win_size, 1)
            window = np.outer(k1d, k1d)
            window /= np.sum(window)

            # 计算均值
            mu1 = signal.convolve2d(ref, window, mode='valid')
            mu2 = signal.convolve2d(dist, window, mode='valid')

            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            # 计算方差和协方差
            sigma1_sq = signal.convolve2d(ref * ref, window, mode='valid') - mu1_sq
            sigma2_sq = signal.convolve2d(dist * dist, window, mode='valid') - mu2_sq
            sigma12 = signal.convolve2d(ref * dist, window, mode='valid') - mu1_mu2

            # 数值保护
            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            # 估计增益因子 g
            g = sigma12 / (sigma1_sq + eps)

            # 估计噪声方差 sv_sq
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = 0

            sv_sq[sv_sq < eps] = eps

            # 计算 VIF
            # 分子: 失真图像的信息量
            num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))
            # 分母: 参考图像的信息量
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

            if den == 0: return 0
            return num / den

        # 计算融合图像相对于两幅源图像的 VIF 之和
        return vif_single(a, f) + vif_single(b, f)

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

        return correlation(diff_a, b) + correlation(diff_b, a)

    def Qabf(self, fused, img_a, img_b):
        """
        Qabf (Edge Preservation Index) - 最终融合修正版
        """

        # 【关键修正】：不要使用 min-max 动态拉伸，而是统一线性缩放
        # 假设输入是 [0, 1] 的 tensor 或 numpy
        def safe_scale(img):
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            if img.ndim == 4:
                img = img[0, 0]
            elif img.ndim == 3:
                img = img[0]
            # 这里的 255.0 是为了让梯度数值在常见范围内，不改变对比度
            return img.astype(np.float32) * 255.0

        f = safe_scale(fused)
        a = safe_scale(img_a)
        b = safe_scale(img_b)

        # 定义 Sobel 核
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        def get_gradient(img):
            gx = signal.convolve2d(img, sobel_x, mode='same')
            gy = signal.convolve2d(img, sobel_y, mode='same')
            g_strength = np.sqrt(gx ** 2 + gy ** 2)
            g_alpha = np.arctan2(gy, gx)  # 角度 [-pi, pi]
            return g_strength, g_alpha

        g_a, alpha_a = get_gradient(a)
        g_b, alpha_b = get_gradient(b)
        g_f, alpha_f = get_gradient(f)

        # Qabf 标准参数 (Xydeas & Petrovic 2000)
        L, Tg, kg, Dg, Ta, ka, Da = 1, 0.9994, -15, 0.5, 0.9879, -22, 0.8

        def edge_preservation(g1, a1, g2, a2):
            # --- 1. 强度保留值 Qg ---
            denom = np.maximum(g1, g2)
            denom[denom == 0] = 1e-6
            # 使用 min/max 计算比率 (0~1)
            G_sim = np.minimum(g1, g2) / denom
            Qg = Tg / (1 + np.exp(kg * (G_sim - Dg)))

            # --- 2. 角度保留值 Qa ---
            diff_alpha = np.abs(a1 - a2)
            diff_alpha = np.mod(diff_alpha, 2 * np.pi)
            diff_alpha[diff_alpha > np.pi] = 2 * np.pi - diff_alpha[diff_alpha > np.pi]

            # 将角度差异转化为相似度 (0~1)
            A_sim = 1 - diff_alpha / (np.pi / 2)
            A_sim[A_sim < 0] = 0
            Qa = Ta / (1 + np.exp(ka * (A_sim - Da)))

            return Qg * Qa

        Q_af = edge_preservation(g_a, alpha_a, g_f, alpha_f)
        Q_bf = edge_preservation(g_b, alpha_b, g_f, alpha_f)

        # 权重
        w_a = g_a ** L
        w_b = g_b ** L
        sum_w = w_a + w_b
        sum_w[sum_w == 0] = 1e-6

        numerator = Q_af * w_a + Q_bf * w_b
        Q = np.sum(numerator) / np.sum(sum_w)
        return Q

    def Avg_SSIM(self, fused, img_a, img_b):
        """结构相似性 (仅调用你之前的 pytorch 实现)"""
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