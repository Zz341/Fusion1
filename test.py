import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image  # 用于保存图片

from config import Config
from models.network import CWAF_Net
from utils.dataset import FusionDataset
from utils.metrics import FusionMetrics


def test_and_evaluate():
    # 1. 初始化配置
    cfg = Config()
    device = cfg.device
    print(f"🚀 Start Testing & Evaluation on {device}...")

    # ================= 文件保存路径设置 =================
    # 结果总目录: results/
    # 图片保存: results/fused_images/
    # 报告保存: results/evaluation_report.txt
    output_root = "results"
    image_save_dir = os.path.join(output_root, "fused_images")
    report_path = os.path.join(output_root, "evaluation_report.txt")

    os.makedirs(image_save_dir, exist_ok=True)
    # ===================================================

    # 2. 加载测试数据
    # 优先找 'test' 目录，没有则用 'train' 代替演示
    test_dataset = FusionDataset(cfg.DATA_ROOT, mode='test', img_size=cfg.img_size)
    if len(test_dataset) == 0:
        print("⚠️ Warning: Test dataset empty, using Train dataset for demo.")
        test_dataset = FusionDataset(cfg.DATA_ROOT, mode='train', img_size=cfg.img_size)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 3. 加载模型
    model = CWAF_Net(in_channels=cfg.in_channels, feat_dim=cfg.feat_dim).to(device)

    # 🔴 请确认这是您想测试的权重文件
    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "cwaf_epoch_100.pth")

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Model loaded from {checkpoint_path}")
    else:
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}!")
        return

    model.eval()

    # 4. 初始化指标计算器
    metrics_calc = FusionMetrics(device=device)

    # 指标结果存储列表
    metric_results = {
        "EN": [], "SD": [], "SF": [], "AG": [],
        "MI": [], "SCD": [], "Qabf": [], "SSIM": []
    }

    print(f"Processing {len(test_dataset)} image pairs...")

    # 开始推理
    with torch.no_grad():
        # 使用 tqdm 显示进度条
        for i, (img_a, img_b, names) in enumerate(tqdm(test_loader, ncols=100)):
            img_a = img_a.to(device)
            img_b = img_b.to(device)

            # --- A. 推理 (Inference) ---
            fused = model(img_a, img_b)
            fused = torch.clamp(fused, 0, 1)  # 确保像素值在有效范围内

            # --- B. 保存图片 (Save Images) ---
            # 假设 dataset 返回的文件名在 names 元组里，如果没有则用索引命名
            file_name = f"{i + 1:03d}_fused.png"
            save_path = os.path.join(image_save_dir, file_name)
            save_image(fused, save_path)

            # --- C. 计算指标 (Calculate Metrics) ---
            metric_results["EN"].append(metrics_calc.EN(fused))
            metric_results["SD"].append(metrics_calc.SD(fused))
            metric_results["SF"].append(metrics_calc.SF(fused))
            metric_results["AG"].append(metrics_calc.AG(fused))
            metric_results["MI"].append(metrics_calc.MI(fused, img_a, img_b))
            metric_results["SCD"].append(metrics_calc.SCD(fused, img_a, img_b))
            metric_results["Qabf"].append(metrics_calc.Qabf(fused, img_a, img_b))
            metric_results["SSIM"].append(metrics_calc.MS_SSIM(fused, img_a, img_b))

    # 5. 整理结果并保存
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 准备报告内容的字符串
    report_lines = []
    report_lines.append(f"==================================================")
    report_lines.append(f"📅 Evaluation Report - {current_time}")
    report_lines.append(f"🤖 Model: {checkpoint_path}")
    report_lines.append(f"🖼️  Test Set Size: {len(test_dataset)}")
    report_lines.append(f"==================================================")
    report_lines.append(f"{'Metric':<10} | {'Average':<10} | {'Std Dev':<10}")
    report_lines.append(f"--------------------------------------------------")

    # 打印到控制台
    print("\n" + "\n".join(report_lines))

    # 遍历指标计算平均值
    for key, val_list in metric_results.items():
        avg_val = np.mean(val_list)
        std_val = np.std(val_list)
        line = f"{key:<10} | {avg_val:<10.4f} | {std_val:<10.4f}"
        print(line)
        report_lines.append(line)

    print("==================================================\n")

    # --- D. 写入 TXT 文件 (Append模式，不覆盖旧记录) ---
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        f.write("\n\n")  # 空两行，方便区分下一次记录

    print(f"🎉 All Done!")
    print(f"   - Images saved to: {image_save_dir}")
    print(f"   - Report saved to: {report_path}")


if __name__ == "__main__":
    test_and_evaluate()