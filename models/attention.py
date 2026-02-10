import torch
import torch.nn as nn


# --- 基础组件 ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# --- 核心模块 ---

class BiDirectionalCrossAttention(nn.Module):
    """用于低频特征的全局交互"""

    def __init__(self, dim, num_heads=8):
        super(BiDirectionalCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward_single(self, x_q, x_kv):
        B, C, H, W = x_q.shape
        q = x_q.flatten(2).transpose(1, 2)
        k = x_kv.flatten(2).transpose(1, 2)
        v = x_kv.flatten(2).transpose(1, 2)

        q = q.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, -1, C)
        return self.out_proj(out).transpose(1, 2).reshape(B, C, H, W), attn

    def forward(self, feat_a, feat_b):
        out_a2b, attn_a = self.forward_single(feat_a, feat_b)
        out_b2a, attn_b = self.forward_single(feat_b, feat_a)
        fused = torch.cat([out_a2b, out_b2a, feat_a, feat_b], dim=1)
        return fused, (attn_a + attn_b) / 2


class SpatialGatingUnit(nn.Module):
    """用于高频特征的硬/软门控选择 - 修复版"""

    def __init__(self, channels):
        super(SpatialGatingUnit, self).__init__()

        # 决策网络：输入 concat 的特征，输出掩膜 (Mask)
        # 注意：这里处理的是高频系数，必须保留符号信息
        self.decision_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),  # 使用 LeakyReLU 保留负信号
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 特征变换：仅做线性变换或轻微非线性，绝对不要用 ReLU
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),  # Depthwise Conv 保留独立性
            nn.GroupNorm(4, channels),
            # 移除 CBAM，因为它包含 ReLU 和 AvgPool，不适合高频小波系数
        )

    def forward(self, h_a, h_b):
        # 1. 生成决策 Mask
        cat_feat = torch.cat([h_a, h_b], dim=1)
        mask = self.decision_conv(cat_feat)

        # 2. 测试时使用硬阈值 (Hard Selection) 以最大化 Qabf
        if not self.training:
            mask = (mask > 0.5).float()

        # 3. 融合 (软/硬 门控)
        h_fused = h_a * mask + h_b * (1 - mask)

        # 4. 特征变换 (可选，如果效果不好可直接返回 h_fused)
        out = self.transform(h_fused)

        # 残差连接：防止变换丢失信息
        return out + h_fused


class RRB(nn.Module):
    """残差细化块"""

    def __init__(self, channels):
        super(RRB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.2, inplace=True),  # 使用 LeakyReLU 避免死区
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels)
        )
        self.cbam = CBAM(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)  # 输出也用 LeakyReLU

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.cbam(out)
        out += residual
        return self.act(out)