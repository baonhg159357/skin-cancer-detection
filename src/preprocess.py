import os
import numpy as np
import pandas as pd
import cv2

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


class HAM10000Preprocessor:
    def __init__(self, raw_csv_path: str, output_dir: str, raw_img_dir: str = None):
        self.raw_csv_path = raw_csv_path
        self.output_dir = output_dir
        self.raw_img_dir = raw_img_dir  # Thư mục chứa các folder ảnh gốc (part_1, part_2)
        self.df = None

    def load_data(self) -> 'HAM10000Preprocessor':
        """
            Đọc file metadata gốc từ thư mục raw.
            Nếu file không tồn tại, sẽ ném ra lỗi FileNotFoundError.
        """
        if not os.path.exists(self.raw_csv_path):
            raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {self.raw_csv_path}")
        
        self.df = pd.read_csv(self.raw_csv_path)
        print(f"Đã tải thành công metadata với {len(self.df)} dòng dữ liệu.")
        return self

    @staticmethod
    def crop_lesion_roi(image: np.ndarray) -> np.ndarray:
        """
        Cắt bỏ viền đen xung quanh ảnh dermoscopy và tập trung vào vùng tổn thương trung tâm (ROI).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Ngưỡng hóa để loại bỏ viền đen ngoài biên (pixel có giá trị quá thấp coi như nền đen)
            _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            
            # Tìm các contours trên ảnh nhị phân
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return image
            
            # Lấy contour lớn nhất (vùng chứa tổn thương chính)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Kiểm tra nếu diện tích vùng crop hợp lý (tránh trường hợp contour chiếm toàn bộ ảnh hoặc quá nhỏ)
            img_h, img_w = image.shape[:2]
            if w > int(img_w * 0.3) and h > int(img_h * 0.3):
                cropped = image[y:y+h, x:x+w]
                return cropped
        except Exception:
            pass
            
        return image

    @staticmethod
    def remove_hair(image: np.ndarray) -> np.ndarray:
        """
            Thuật toán DullRazor thu gọn để loại bỏ tóc/lông che khuất trên ảnh da liễu.
        """
        grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        cleaned_image = cv2.inpaint(image, thresh, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cleaned_image

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """
            Áp dụng cân bằng độ tương phản cục bộ CLAHE trên không gian màu LAB.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return enhanced_image

    def preprocess_images(self) -> None:
        """Quét toàn bộ ảnh gốc, thực hiện Crop ROI + Khử lông + CLAHE và lưu sang thư mục processed images."""
        if not self.raw_img_dir or not os.path.exists(self.raw_img_dir):
            print("Cảnh báo: Không tìm thấy thư mục ảnh gốc (`raw_img_dir`). Bỏ qua bước tiền xử lý ảnh.")
            return

        processed_img_output_dir = os.path.join(self.output_dir, "images_processed")
        os.makedirs(processed_img_output_dir, exist_ok=True)
        
        print("Đang tiến hành tiền xử lý nâng cao (Crop ROI + Khử lông + CLAHE) cho toàn bộ ảnh...")
        count = 0
        
        for root, _, files in os.walk(self.raw_img_dir):
            for file in files:
                if file.endswith(".jpg"):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(processed_img_output_dir, file)
                    
                    if not os.path.exists(dst_path):
                        try:
                            image = cv2.imread(src_path)
                            if image is None:
                                continue
                            
                            # 1. Cắt bỏ viền đen / lấy vùng tổn thương trung tâm (Crop ROI)
                            cropped_image = self.crop_lesion_roi(image)
                            
                            # 2. Khử lông (DullRazor)
                            hair_removed = self.remove_hair(cropped_image)
                            
                            # 3. Tăng cường tương phản cục bộ (CLAHE)
                            final_processed = self.apply_clahe(hair_removed)
                            
                            cv2.imwrite(dst_path, final_processed)
                            count += 1
                        except Exception as e:
                            print(f"Lỗi khi xử lý ảnh {file}: {e}")
                            
        print(f"Đã hoàn tất tiền xử lý cho {count} bức ảnh -> Lưu tại: {processed_img_output_dir}")

    def split_data(self, train_size: float = 0.7, val_size: float = 0.15) -> tuple:
        """
        Phân chia dữ liệu theo patient_id/lesion_id để tránh rò rỉ dữ liệu (data leakage).
        """
        if self.df is None:
            raise ValueError("Dữ liệu chưa được tải. Hãy gọi phương thức load_data() trước.")

        gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
        train_idx, temp_idx = next(gss1.split(self.df, groups=self.df['lesion_id']))
        
        train_df = self.df.iloc[train_idx].reset_index(drop=True)
        temp_df = self.df.iloc[temp_idx].reset_index(drop=True)
        
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
        """Thực thi toàn bộ pipeline tiền xử lý (Crop ROI + Khử nhiễu + CLAHE + Phân chia dữ liệu)."""
        self.load_data()
        self.preprocess_images()
        train_df, val_df, test_df = self.split_data()
        self.save_splits(train_df, val_df, test_df)

if __name__ == "__main__":
    # Tự động lấy thư mục gốc của project (lùi 1 cấp từ thư mục src)
    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    preprocessor = HAM10000Preprocessor(
        raw_csv_path=str(ROOT_DIR / "data" / "raw" / "HAM10000_metadata"),
        output_dir=str(ROOT_DIR / "data" / "processed"),
        raw_img_dir=str(ROOT_DIR / "data" / "raw")
    )
    preprocessor.run()