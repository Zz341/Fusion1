import os
import sys  # 引入 sys 用于解决控制台输出流问题

# [优化] 1. 显存碎片管理 (必须放在 import torch 之前)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# [优化] 2. 导入混合精度训练库
from torch.cuda.amp import autocast, GradScaler

from config import Config
from models.network import CWAF_Net
from utils.loss import FusionLoss
from utils.dataset import FusionDataset


def train():
    # 1. 初始化配置
    cfg = Config()
    cfg.create_dirs()
    print(f"Start Training on {cfg.device}...")

    # 2. 准备数据
    # 注意: 在 Windows 上如果报错内存不足，请确保 config.py 中 num_workers=0
    train_dataset = FusionDataset(cfg.DATA_ROOT, mode='train', img_size=cfg.img_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"Training data loaded: {len(train_dataset)} pairs.")

    # 3. 初始化模型、损失函数、优化器
    model = CWAF_Net(in_channels=cfg.in_channels, feat_dim=cfg.feat_dim).to(cfg.device)
    criterion = FusionLoss(
        device=cfg.device,
        int_w=cfg.lambda_int,
        grad_w=cfg.lambda_grad,
        ssim_w=cfg.lambda_ssim
    )
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    # 学习率策略 (按 Epoch 更新)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

    # [优化] 3. 初始化混合精度 Scaler
    scaler = GradScaler()

    # 4. 训练循环
    model.train()
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0

        # =================================================================================
        # [核心修改] 进度条配置
        # leave=False: 跑完一轮后进度条自动消失，不占用屏幕空间
        # file=sys.stdout: 强制输出到标准流，防止 PyCharm 刷屏
        # ncols=100: 固定宽度，防止自动换行
        # =================================================================================
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}", file=sys.stdout, ncols=100,
                  leave=False) as progress_bar:
            for i, (img_a, img_b, _) in enumerate(progress_bar):
                img_a = img_a.to(cfg.device)
                img_b = img_b.to(cfg.device)

                optimizer.zero_grad()

                # [优化] 4. 开启混合精度上下文
                with autocast():
                    fused_img = model(img_a, img_b)
                    loss = criterion(fused_img, img_a, img_b)

                # [优化] 5. 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

                # 更新进度条上的瞬时 Loss
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # =================================================================================
        # 退出 with 循环后，进度条自动清除。
        # 此时打印 Summary，界面会非常清爽。
        # =================================================================================

        # 1. 计算平均 Loss
        avg_loss = epoch_loss / len(train_loader)

        # 2. 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 3. 打印整齐的日志
        print(f"Epoch [{epoch + 1}/{cfg.num_epochs}] Average Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

        # 4. 更新学习率
        scheduler.step()

        # 5. 保存模型 (每10轮保存一次，以及最后一轮)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.num_epochs:
            save_path = os.path.join(cfg.CHECKPOINT_DIR, f"cwaf_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            # print(f" Model saved to {save_path}") # 可选：为了保持清爽，可以注释掉这行

    print("Training Finished!")


if __name__ == "__main__":
    try:
        train()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("\n" + "=" * 40)
            print("❌ 显存不足 (OOM) 警告")
            print("请检查 config.py 中的 batch_size 是否已改为 1")
            print("=" * 40 + "\n")
        raise e