import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# 引入你的模块
from config import Config
from models.network import CWAF_Net
from utils.loss import FusionLoss


def load_image(image_path, img_size=256):
    """
    读取单张图片并转换为模型输入的Tensor格式
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # 定义预处理: 转灰度 -> 调整大小 -> 转Tensor -> 归一化
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 强制转为单通道灰度图
        transforms.Resize((img_size, img_size)),  # 调整到 Config 中定义的大小
        transforms.ToTensor(),  # 转为 [0, 1] 的 Tensor
        # transforms.Normalize(mean=[0.5], std=[0.5]) # 可选: 归一化到 [-1, 1]
    ])

    img = Image.open(image_path)
    img_tensor = transform(img)

    # 增加 Batch 维度: [C, H, W] -> [1, C, H, W]
    return img_tensor.unsqueeze(0)


def save_image(tensor, save_path):
    """
    将输出Tensor保存为图片
    """
    # 如果模型输出包含Tanh，范围是[-1, 1]，需要反归一化
    # 这里假设输出直接是像素强度或特征映射
    tensor = tensor.squeeze().cpu().detach()

    # 简单的截断处理，防止像素溢出
    tensor = torch.clamp(tensor, 0, 1)

    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    img.save(save_path)
    print(f"Fused image saved to: {save_path}")


def main():
    # 1. 初始化配置
    cfg = Config()
    print(f"Running on device: {cfg.device}")

    # 2. 实例化模型
    model = CWAF_Net(in_channels=cfg.in_channels, feat_dim=cfg.feat_dim).to(cfg.device)
    print("Model initialized.")

    # 3. 加载真实图像 (而不是随机噪声)
    # ==========================================
    # 这里的路径需要在 config.py 中修改，或者直接在这里写死测试
    # 为了演示方便，这里先检查文件是否存在，不存在则生成随机图
    if os.path.exists(cfg.TEST_IMG_A_PATH) and os.path.exists(cfg.TEST_IMG_B_PATH):
        print(f"Loading images from:\n A: {cfg.TEST_IMG_A_PATH}\n B: {cfg.TEST_IMG_B_PATH}")
        img_a = load_image(cfg.TEST_IMG_A_PATH, cfg.img_size).to(cfg.device)
        img_b = load_image(cfg.TEST_IMG_B_PATH, cfg.img_size).to(cfg.device)
    else:
        print("[Warning] Test images not found in config paths. Using random noise for demo.")
        img_a = torch.randn(1, 1, cfg.img_size, cfg.img_size).to(cfg.device)
        img_b = torch.randn(1, 1, cfg.img_size, cfg.img_size).to(cfg.device)
    # ==========================================

    # 4. 前向传播 (推理)
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        print("Start Fusion...")
        fused_img = model(img_a, img_b)

    print(f"Input Shape: {img_a.shape}")
    print(f"Output Shape: {fused_img.shape}")

    # 5. 如果是随机噪声，计算损失演示一下
    # 如果是真实推理，通常不需要计算损失，除非你有Ground Truth
    criterion = FusionLoss(device=cfg.device)
    loss = criterion(fused_img, img_a, img_b)
    print(f"Reconstruction Loss (Unsupervised): {loss.item()}")

    # 6. 保存结果
    save_path = os.path.join(cfg.RESULT_DIR, 'fused_result.png')
    save_image(fused_img, save_path)

    # 7. 可视化对比 (可选)
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1); plt.imshow(img_a.squeeze().cpu(), cmap='gray'); plt.title('Source A')
    # plt.subplot(1, 3, 2); plt.imshow(img_b.squeeze().cpu(), cmap='gray'); plt.title('Source B')
    # plt.subplot(1, 3, 3); plt.imshow(fused_img.squeeze().cpu(), cmap='gray'); plt.title('Fused')
    # plt.show()


if __name__ == "__main__":
    main()