import os
import numpy as np
import sys
from pathlib import Path
from loss import FocalLoss
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader


# Thêm thư mục gốc (root) vào hệ thống path để Python có thể import được config.py từ ngoài vào
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import các hằng số và đường dẫn tập trung từ config ở thư mục gốc
from config import (
    RAW_DATA_DIR,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    BEST_MODEL_PATH,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    MODEL_DIR
)

# Import các module đã xây dựng trong dự án
from dataset import SkinCancerDataset, get_albumentations_transforms
from model import get_skin_cancer_model

def compute_smoothed_class_weights(train_csv_path: str, num_classes: int = 7, beta: float = 0.5) -> torch.Tensor:
    """
    Tính toán trọng số lớp được làm mềm (Smoothed Class Weights) 
    bằng cách sử dụng hệ số mũ beta (mặc định 0.5 tương đương căn bậc hai)
    nhằm tránh việc phóng đại trọng số quá mức ở các lớp thiểu số.
    """
    df = pd.read_csv(train_csv_path)
    class_counts = df['dx'].value_counts().sort_index().values
    
    # Công thức làm mềm: Nghịch đảo tần suất được đưa về lũy thừa beta
    # beta = 0.5 giúp giảm bớt khoảng cách chênh lệch lớn giữa lớp đa số và thiểu số
    weights = 1.0 / (class_counts ** beta)
    
    # Chuẩn hóa để tổng trọng số cân đối
    weights = weights / np.sum(weights) * num_classes
    
    class_weights = torch.FloatTensor(weights)
    print(f"Đã tính toán xong Smoothed Class Weights (beta={beta}): {class_weights}")
    return class_weights

def train_model():
    # 1. Cấu hình thiết bị (Tự động chọn GPU nếu khả dụng, ngược lại dùng CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    if device.type == "cuda":
        print(f"- GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Sử dụng trực tiếp các đường dẫn và tham số từ file config tập trung
    print(f"- Data Directory: {RAW_DATA_DIR}")
    print(f"- Train CSV: {TRAIN_CSV_PATH}")
    print(f"- Val CSV: {VAL_CSV_PATH}")

    # 3. Khởi tạo Transforms và Datasets
    train_transform, val_transform = get_albumentations_transforms()
    
    train_dataset = SkinCancerDataset(csv_file=TRAIN_CSV_PATH, img_dir=RAW_DATA_DIR, transform=train_transform)
    val_dataset = SkinCancerDataset(csv_file=VAL_CSV_PATH, img_dir=RAW_DATA_DIR, transform=val_transform)
    
    # Tạo DataLoaders để nạp dữ liệu theo từng batch
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Loaded training samples: {len(train_dataset)}")
    print(f"Loaded validation samples: {len(val_dataset)}")

    # 4. Khởi tạo mô hình, tính toán trọng số Loss và thuật toán tối ưu
    # Sử dụng freeze_backbone=True ở các epoch đầu để giữ vững kiến thức nền tảng
    model = get_skin_cancer_model(model_name="resnet50", num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
    model = model.to(device)
    
    # 5. Tính smoothed class weights làm alpha cho Focal Loss
    class_weights = compute_smoothed_class_weights(TRAIN_CSV_PATH, num_classes=NUM_CLASSES, beta=0.5).to(device)

    # 6. Khởi tạo Focal Loss với gamma = 2.0
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    # 5. Vòng lặp huấn luyện chính (Training Loop)
    best_val_acc = 0.0
    
    # Đảm bảo thư mục lưu model tồn tại
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch [{epoch+1}/{NUM_EPOCHS}] ---")
        
        # --- Pha Huấn Luyện (Training Phase) ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # --- Pha Đánh Giá (Validation Phase) ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}%")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc * 100:.2f}%")
        
        # Lưu lại checkpoint tốt nhất sử dụng đường dẫn chuẩn hóa từ config.py
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"--> Saved new best model to {BEST_MODEL_PATH} with Val Acc: {best_val_acc * 100:.2f}%")

if __name__ == "__main__":
    train_model()