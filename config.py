import os
from pathlib import Path

# 1. Vì file config.py nằm ngay ở thư mục gốc, ROOT_DIR chính là thư mục hiện tại của file này
ROOT_DIR = Path(__file__).resolve().parent

# 2. Cấu trúc các đường dẫn thư mục dữ liệu
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 3. Đường dẫn chi tiết đến các file CSV đã xử lý
TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_CSV_PATH = PROCESSED_DATA_DIR / "test.csv"

# 4. Thư mục và file lưu trữ mô hình (Model Checkpoints)
MODEL_DIR = ROOT_DIR / "saved_models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"

# 5. Các siêu tham số (Hyperparameters) dùng chung cho toàn dự án
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 7
IMAGE_SIZE = (224, 224)

# Tự động tạo sẵn các thư mục quan trọng nếu chưa tồn tại
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)