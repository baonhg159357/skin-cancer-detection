import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

class HAM10000Preprocessor:
    def __init__(self, raw_csv_path: str, output_dir: str):
        self.raw_csv_path = raw_csv_path
        self.output_dir = output_dir
        self.df = None

    def load_data(self) -> 'HAM10000Preprocessor':
        """Đọc file metadata gốc từ thư mục raw."""
        if not os.path.exists(self.raw_csv_path):
            raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {self.raw_csv_path}")
        
        self.df = pd.read_csv(self.raw_csv_path)
        print(f"Đã tải thành công metadata với {len(self.df)} dòng dữ liệu.")
        return self

    def split_data(self, train_size: float = 0.7, val_size: float = 0.15) -> tuple:
        """
        Phân chia dữ liệu theo patient_id/lesion_id để tránh rò rỉ dữ liệu (data leakage).
        """
        if self.df is None:
            raise ValueError("Dữ liệu chưa được tải. Hãy gọi phương thức load_data() trước.")

        # 1. Chia tập Train và phần còn lại (Temp)
        gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
        train_idx, temp_idx = next(gss1.split(self.df, groups=self.df['lesion_id']))
        
        train_df = self.df.iloc[train_idx].reset_index(drop=True)
        temp_df = self.df.iloc[temp_idx].reset_index(drop=True)
        
        # 2. Chia phần Temp thành Validation và Test (tỷ lệ 50-50 cho phần còn lại)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
        val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['lesion_id']))
        
        val_df = temp_df.iloc[val_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)
        
        return train_df, val_df, test_df

    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Lưu các tập dữ liệu đã chia vào thư mục processed."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        train_path = os.path.join(self.output_dir, 'train.csv')
        val_path = os.path.join(self.output_dir, 'val.csv')
        test_path = os.path.join(self.output_dir, 'test.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print("Đã hoàn tất phân chia và lưu file:")
        print(f"  - Train: {len(train_df)} mẫu -> {train_path}")
        print(f"  - Validation: {len(val_df)} mẫu -> {val_path}")
        print(f"  - Test: {len(test_df)} mẫu -> {test_path}")

    def run(self) -> None:
        """Thực thi toàn bộ pipeline tiền xử lý."""
        self.load_data()
        train_df, val_df, test_df = self.split_data()
        self.save_splits(train_df, val_df, test_df)

if __name__ == "__main__":
    preprocessor = HAM10000Preprocessor(
        raw_csv_path="../data/raw/HAM10000_metadata",
        output_dir="../data/processed/"
    )
    preprocessor.run()