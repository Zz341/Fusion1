import torch
import torch.nn as nn
from models.dwt import DWT_Fixed, IDWT_Fixed
from models.backbone import ResNetBackbone
from models.attention import BiDirectionalCrossAttention, SpatialGatingUnit, RRB

class CWAF_Net(nn.Module):
    def __init__(self, in_channels=1, feat_dim=64):
        super(CWAF_Net, self).__init__()

        self.backbone = ResNetBackbone(in_channels, feat_dim)
        self.dwt = DWT_Fixed()
        self.idwt = IDWT_Fixed()

        # 低频处理
        self.low_freq_attn = BiDirectionalCrossAttention(feat_dim)
        self.low_freq_fusion = nn.Conv2d(feat_dim * 4, feat_dim, 1)

        # 高频处理 (3个方向)
        self.gate_l2 = nn.ModuleList([SpatialGatingUnit(feat_dim) for _ in range(3)])
        self.gate_l1 = nn.ModuleList([SpatialGatingUnit(feat_dim) for _ in range(3)])

        # 重构细化
        self.rrb_l2 = RRB(feat_dim)
        self.rrb_l1 = RRB(feat_dim)

        self.res_fusion = nn.Conv2d(feat_dim * 2, feat_dim, 1)

        # 【修改】输出层：不再使用 Sigmoid 直接输出，而是输出残差
        self.final_conv = nn.Sequential(
            nn.Conv2d(feat_dim, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
            # 移除 Sigmoid，放到 forward 最后处理
        )

    def forward(self, img_a, img_b):
        # 1. 提取特征
        f_a = self.backbone(img_a)
        f_b = self.backbone(img_b)

        # 2. 小波分解
        a_ll1, a_high1 = self.dwt(f_a)
        b_ll1, b_high1 = self.dwt(f_b)
        a_ll2, a_high2 = self.dwt(a_ll1)
        b_ll2, b_high2 = self.dwt(b_ll1)

        # 3. 融合
        # 低频
        f_ll2_cat, _ = self.low_freq_attn(a_ll2, b_ll2)
        f_ll2 = self.low_freq_fusion(f_ll2_cat)

        # 高频
        f_high2 = tuple([self.gate_l2[i](a_high2[i], b_high2[i]) for i in range(3)])
        f_high1 = tuple([self.gate_l1[i](a_high1[i], b_high1[i]) for i in range(3)])

        # 4. 重构
        rec_l1 = self.rrb_l2(self.idwt(f_ll2, f_high2))
        rec_feat = self.rrb_l1(self.idwt(rec_l1, f_high1))

        # 5. 残差融合
        res_feat = self.res_fusion(torch.cat([f_a, f_b], dim=1))
        out_feat = rec_feat + res_feat

        # 6. 【修改】生成残差图
        residual_img = self.final_conv(out_feat)

        # 7. 【核心修改】Input Injection 策略
        # 基础图像 = max(A, B) 可以很好地保留骨骼和纹理基础
        base_img = torch.max(img_a, img_b)

        # 最终图像 = 基础 + 网络学习到的细节修正
        out = torch.sigmoid(base_img + residual_img)

        return out