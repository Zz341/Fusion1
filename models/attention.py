import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(BiDirectionalCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # 线性投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward_single_direction(self, x_q, x_kv):
        B, C, H, W = x_q.shape
        # Flatten: [B, H*W, C]
        q = x_q.flatten(2).transpose(1, 2)
        k = x_kv.flatten(2).transpose(1, 2)
        v = x_kv.flatten(2).transpose(1, 2)

        # Reshape q, k, v for Multi-Head Attention
        # [B, N, C] -> [B, N, Heads, C_head] -> [B, Heads, N, C_head]
        q = q.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Attention: [B, Heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Value aggregation
        out = (attn @ v)  # [B, Heads, N, C_head]

        # Merge Heads: [B, Heads, N, C_head] -> [B, N, Heads*C_head]
        out = out.permute(0, 2, 1, 3).reshape(B, -1, C)
        out = self.out_proj(out)

        # Reshape back: [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out, attn

    def forward(self, feat_a, feat_b):
        # 路径1: A 查询 B
        out_a2b, attn_map_a = self.forward_single_direction(feat_a, feat_b)

        # 路径2: B 查询 A
        out_b2a, attn_map_b = self.forward_single_direction(feat_b, feat_a)

        # 融合
        fused = torch.cat([out_a2b, out_b2a, feat_a, feat_b], dim=1)

        # 返回平均注意力图 [B, Heads, N, N]
        return fused, (attn_map_a + attn_map_b) / 2


class SpatialGatingUnit(nn.Module):
    def __init__(self, channels):
        super(SpatialGatingUnit, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_a, h_b, attn_map):
        # h_a, h_b: 当前层级的高频特征 [B, C, H, W] (例如 128x128)
        # attn_map: 来自低频层的注意力图 [B, Heads, N, N] (N 来自 64x64 = 4096)

        B, C, H, W = h_a.shape

        # 1. 对 Heads 维度取平均: [B, Heads, N, N] -> [B, N, N]
        attn_avg = attn_map.mean(dim=1)

        # 2. 获取每个位置的空间权重: [B, N]
        # 对 dim=2 (Key) 求平均，表示 Query 位置 i 收到的平均关注度
        spatial_weight_flat = attn_avg.mean(dim=2)

        # 3. 动态计算源分辨率 (Level 2 的大小)
        # N = 4096 -> sqrt(4096) = 64
        N = spatial_weight_flat.shape[1]
        H_low = int(N ** 0.5)
        W_low = int(N ** 0.5)

        # 4. 还原为 2D 图像: [B, 1, 64, 64]
        spatial_weight = spatial_weight_flat.reshape(B, 1, H_low, W_low)

        # 5. 上采样插值到当前层级的分辨率: [64, 64] -> [128, 128]
        mask = F.interpolate(spatial_weight, size=(H, W), mode='bilinear', align_corners=False)
        mask = self.sigmoid(mask)

        # 融合策略
        h_fused_raw = h_a * mask + h_b * (1 - mask)

        # 卷积整合
        combined = torch.cat([h_fused_raw, h_a + h_b], dim=1)
        out = self.conv(combined)
        return out


class RRB(nn.Module):
    """Residual Refinement Block"""

    def __init__(self, channels):
        super(RRB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return self.relu(out)