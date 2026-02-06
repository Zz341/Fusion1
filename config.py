import os
import torch


class Config:
    # ================= 基础设置 =================
    project_name = "CWAF_Fusion"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

    # ================= 路径设置 =================
    # 假设你的数据结构是:
    # dataset/
    #   ├── train/
    #   │   ├── CT/
    #   │   └── MRI/
    #   └── test/
    #       ├── CT/
    #       └── MRI/

    # 数据集根目录
    DATA_ROOT = './dataset'

    # 训练结果保存路径 (模型权重, 日志)
    CHECKPOINT_DIR = './checkpoints'
    RESULT_DIR = './results'

    # 如果是单次推理(测试)，指定两张具体的图片路径
    TEST_IMG_A_PATH = './dataset/test/MRI/2004.png'
    TEST_IMG_B_PATH = './dataset/test/CT/2004.png'

    # ================= 模型参数 =================
    in_channels = 1  # 输入通道 (灰度图=1, RGB=3)
    feat_dim = 64  # 骨架网络输出特征维度
    n_res_blocks = 2  # 残差细化块的数量 (可选)

    # ================= 训练参数 =================
    img_size = 256  # 输入图像统一调整的大小
    batch_size = 1
    num_epochs = 100
    learning_rate = 1e-4
    num_workers = 0  # 数据加载线程数 win系统最好设为单线程

    # ================= 损失函数权重 =================
    lambda_int = 1.0  # 强度损失: 保持 1.0，保证亮度正确
    # [重点修改] 梯度损失: 原来 2.0 太小了。
    # 梯度值通常很小(1e-3级别)，需要乘一个大系数才能对梯度产生足够的影响。
    # 建议改为 20.0 甚至 50.0，强迫网络去“扣细节”
    lambda_grad = 20.0
    # [建议修改] SSIM损失: SSIM 对结构很敏感，可以适当调高
    lambda_ssim = 5.0

    # ================= 自动创建目录 =================
    @staticmethod
    def create_dirs():
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.RESULT_DIR, exist_ok=True)


# 初始化时自动创建目录
Config.create_dirs()