import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SkinCancerDataset(Dataset):
    """
    Custom Dataset cho bộ dữ liệu HAM10000, nạp ảnh đã qua tiền xử lý (khử lông + CLAHE) 
    và tích hợp Albumentations động.
    """
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Tạo từ điển ánh xạ image_id sang đường dẫn file ảnh
        self.img_paths = self._find_image_paths()

    def _find_image_paths(self) -> dict:
        """Quét và lập bản đồ (mapping) giữa image_id và đường dẫn tuyệt đối của file ảnh."""
        img_path_dict = {}
        if not os.path.exists(self.img_dir):
            return img_path_dict
            
        for root, _, files in os.walk(self.img_dir):
            for file in files:
                if file.endswith(".jpg"):
                    img_id = os.path.splitext(file)[0]
                    img_path_dict[img_id] = os.path.join(root, file)
        return img_path_dict

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # Lấy dòng thông tin theo chỉ số idx
        row = self.df.iloc[idx]
        img_id = row['image_id']
        
        # Tìm đường dẫn ảnh
        img_path = self.img_paths.get(img_id)
        if not img_path:
            raise FileNotFoundError(f"Không tìm thấy file ảnh cho image_id: {img_id}")
        
        # Đọc ảnh bằng OpenCV (BGR) và chuyển sang chuẩn RGB
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Không thể đọc file ảnh tại: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Áp dụng từ điển ánh xạ nhãn 7 lớp bệnh lý của HAM10000
        label_mapping = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
        dx_label = row['dx']
        label = label_mapping.get(dx_label, 0)

        # Áp dụng Albumentations Transform (nếu có) đúng chuẩn từ điển đầu vào
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


def get_albumentations_transforms(image_size: int = 224):
    """
    Định nghĩa các pipeline tăng cường dữ liệu nâng cao bằng Albumentations:
    - Train: Xoay, lật, chỉnh màu, CoarseDropout (chống học vẹt).
    - Val/Test: Resize và chuẩn hóa chuẩn.
    """
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        
        # CoarseDropout (Cutout) chống học vẹt
        A.CoarseDropout(num_holes_range=(8, 12), hole_height_range=(16, 20), hole_width_range=(16, 20), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform