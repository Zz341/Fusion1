import torch
import torch.nn as nn
from .dwt import DWT_Fixed, IDWT_Fixed
from .backbone import ResNetBackbone
from .attention import BiDirectionalCrossAttention, SpatialGatingUnit, RRB


class CWAF_Net(nn.Module):
    def __init__(self, in_channels=1, feat_dim=64):
        super(CWAF_Net, self).__init__()

        # 1. 骨架网络 (现在输出原尺寸 H x W)
        self.backbone = ResNetBackbone(in_channels, feat_dim)

        # 2. 小波变换与逆变换工具
        self.dwt = DWT_Fixed()
        self.idwt = IDWT_Fixed()

        # 3. 跨模态融合模块 (结构不变)
        # Level 1 shape: H/2, W/2 (因为DWT第一层就会减半)
        # Level 2 shape: H/4, W/4

        # 低频融合 (Deepest Level)
        self.low_freq_attn = BiDirectionalCrossAttention(feat_dim)
        self.low_freq_fusion_conv = nn.Conv2d(feat_dim * 4, feat_dim, 1)

        # 高频融合 (Spatial Gating)
        self.gate_l2 = nn.ModuleList([SpatialGatingUnit(feat_dim) for _ in range(3)])
        self.gate_l1 = nn.ModuleList([SpatialGatingUnit(feat_dim) for _ in range(3)])

        # 4. 残差细化块 (RRB)
        self.rrb_l2 = RRB(feat_dim)
        self.rrb_l1 = RRB(feat_dim)

        # --- 修改部分开始: 最终输出层 ---
        # 之前的骨架网络输出是 1/4 大小，现在是原图大小。
        # 经过两层 DWT 后，最深层是 1/4。
        # 经过两层 IDWT 逆变换回来后，rec_feat 的尺寸已经回到了原图大小 (H, W)。
        # 所以这里不需要 ConvTranspose2d (上采样)，只需要 Conv2d (特征映射)。

        self.final_output_layer = nn.Sequential(
            nn.Conv2d(feat_dim, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()  # ✅ 修改为 Sigmoid，将输出限制在 0~1 之间
        )
        # --- 修改部分结束 ---

    def forward(self, img_a, img_b):
        # --- Stage 1: Feature Extraction ---
        # 输出尺寸: [B, 64, H, W] (原尺寸，细节拉满)
        f_a = self.backbone(img_a)
        f_b = self.backbone(img_b)

        # --- Stage 2: Multi-Scale Wavelet Decomposition ---
        # Level 1 (H -> H/2)
        a_ll1, a_high1 = self.dwt(f_a)
        b_ll1, b_high1 = self.dwt(f_b)

        # Level 2 (H/2 -> H/4)
        a_ll2, a_high2 = self.dwt(a_ll1)
        b_ll2, b_high2 = self.dwt(b_ll1)

        # --- Stage 3: Cross-Modal Fusion ---
        # 3.1 低频双向注意力
        f_ll2_concat, attn_map = self.low_freq_attn(a_ll2, b_ll2)
        f_ll2 = self.low_freq_fusion_conv(f_ll2_concat)

        # 3.2 高频空间门控
        f_high2 = []
        for i in range(3):
            f_h = self.gate_l2[i](a_high2[i], b_high2[i], attn_map)
            f_high2.append(f_h)
        f_high2 = tuple(f_high2)

        f_high1 = []
        for i in range(3):
            f_h = self.gate_l1[i](a_high1[i], b_high1[i], attn_map)
            f_high1.append(f_h)
        f_high1 = tuple(f_high1)

        # --- Stage 4: Inverse Reconstruction ---
        # Reconstruct Level 2 -> Level 1 (H/4 -> H/2)
        rec_l1 = self.idwt(f_ll2, f_high2)
        rec_l1 = self.rrb_l2(rec_l1)

        # Reconstruct Level 1 -> Original (H/2 -> H)
        rec_feat = self.idwt(rec_l1, f_high1)
        rec_feat = self.rrb_l1(rec_feat)

        # --- Stage 5: Final Output ---
        # 此时 rec_feat 是 [B, 64, H, W]

        # [改进思路] 把最开始的骨架特征 f_a 和 f_b 加回来
        # 这一步叫 "残差学习"，让网络只需要学习"如何融合"，而不是"如何重画整张图"
        residual_feat = f_a + f_b
        out_feat = rec_feat + residual_feat  # 直接相加

        final_img = self.final_output_layer(out_feat)
        return final_img