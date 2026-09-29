import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SkinCancerDataset(Dataset):
    """
    Custom Dataset cho bộ dữ liệu HAM10000, hỗ trợ nạp ảnh và nhãn động từ file CSV.
    """
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Tạo từ điển ánh xạ image_id sang tên file hoặc tìm kiếm trực tiếp
        # HAM10000 lưu ảnh trong các thư mục con sau khi giải nén
        self.img_paths = self._find_image_paths()

    def _find_image_paths(self) -> dict:
        """Quét và lập bản đồ (mapping) giữa image_id và đường dẫn tuyệt đối của file ảnh."""
        img_path_dict = {}
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
        
        # Mở ảnh bằng thư viện PIL và chuyển sang chuẩn RGB
        image = Image.open(img_path).convert('RGB')
        
        # Lấy nhãn bệnh lý (mã hóa các lớp dx thành số nguyên hoặc giữ nguyên tùy theo chiến lược)
        # HAM10000 có 7 lớp: nv, mel, bkl, bcc, akiec, vasc, df
        label_mapping = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
        dx_label = row['dx']
        label = label_mapping.get(dx_label, 0)

        # Áp dụng các phép biến đổi (Transform / Augmentation) nếu có
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """Định nghĩa các cấu hình tăng cường dữ liệu cho Train và Val/Test."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Chuẩn ImageNet
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform