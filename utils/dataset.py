import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FusionDataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=256):
        """
        root_dir: 数据集根目录 (例如 ./dataset)
        mode: 'train' 或 'test'
        """
        super(FusionDataset, self).__init__()
        self.mode = mode

        # 设定 MRI 和 CT 的文件夹路径
        # 假设结构是: dataset/train/MRI 和 dataset/train/CT
        self.mri_dir = os.path.join(root_dir, mode, 'MRI')
        self.ct_dir = os.path.join(root_dir, mode, 'CT')

        # 确保目录存在，防止报错
        if not os.path.exists(self.mri_dir) or not os.path.exists(self.ct_dir):
            raise FileNotFoundError(f"Data directory not found. Expected: {self.mri_dir} and {self.ct_dir}")

        # 获取文件名列表并排序，确保 MRI 和 CT 一一对应
        self.file_list = sorted(
            [f for f in os.listdir(self.mri_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

        # 图像预处理: 转灰度 -> 调整大小 -> 转Tensor
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]

        mri_path = os.path.join(self.mri_dir, file_name)
        ct_path = os.path.join(self.ct_dir, file_name)

        # 打开图片
        try:
            mri_img = Image.open(mri_path)
            ct_img = Image.open(ct_path)
        except Exception as e:  # 捕获所有异常，不仅仅是 FileNotFoundError
            print(f"⚠️ Error loading {file_name}: {e}")
            # 策略：尝试下一张，如果转了一圈都不行就报错
            new_index = (index + 1) % len(self.file_list)
            if new_index == index:  # 数据集里只有一张图且坏了
                raise e
            return self.__getitem__(new_index)

        # 应用预处理
        mri_tensor = self.transform(mri_img)
        ct_tensor = self.transform(ct_img)

        return mri_tensor, ct_tensor, file_name