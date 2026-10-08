import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Thêm thư mục gốc vào system path để import config dễ dàng
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    RAW_DATA_DIR,
    TEST_CSV_PATH,
    BEST_MODEL_PATH,
    BATCH_SIZE,
    NUM_CLASSES
)
from dataset import SkinCancerDataset, get_albumentations_transforms
from model import get_skin_cancer_model

def evaluate_model():
    # 1. Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 2. Nạp tập Test từ config
    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"Không tìm thấy file test tại: {TEST_CSV_PATH}")

    _, val_transform = get_albumentations_transforms()
    test_dataset = SkinCancerDataset(csv_file=TEST_CSV_PATH, img_dir=RAW_DATA_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Loaded test samples: {len(test_dataset)}")

    # 3. Nạp mô hình đã lưu từ checkpoint tốt nhất
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy checkpoint mô hình tại: {BEST_MODEL_PATH}")

    model = get_skin_cancer_model(model_name="resnet50", num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 4. Tiến hành dự đoán trên toàn bộ tập Test
    all_preds = []
    all_labels = []

    print("Đang chạy đánh giá trên tập Test...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 5. Tính toán các chỉ số đánh giá chuyên sâu
    class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ===")
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")

    print("Classification Report (Precision, Recall/Sensitivity, F1-Score):")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__ == "__main__":
    evaluate_model()