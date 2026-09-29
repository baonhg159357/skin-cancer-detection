import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import các module đã xây dựng trong dự án
from dataset import SkinCancerDataset, get_transforms
from model import get_skin_cancer_model

def train_model():
    # 1. Cấu hình thiết bị (Tự động chọn GPU nếu khả dụng, ngược lại dùng CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    if device.type == "cuda":
        print(f"- GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Định nghĩa các đường dẫn và tham số huấn luyện
    data_dir = "../data/raw"  # Thư mục chứa các folder ảnh đã giải nén
    train_csv = "../data/processed/train.csv"
    val_csv = "../data/processed/val.csv"
    
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    num_classes = 7
    
    # 3. Khởi tạo Transforms và Datasets
    train_transform, val_transform = get_transforms()
    
    train_dataset = SkinCancerDataset(csv_file=train_csv, img_dir=data_dir, transform=train_transform)
    val_dataset = SkinCancerDataset(csv_file=val_csv, img_dir=data_dir, transform=val_transform)
    
    # Tạo DataLoaders để nạp dữ liệu theo từng batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Loaded training samples: {len(train_dataset)}")
    print(f"Loaded validation samples: {len(val_dataset)}")

    # 4. Khởi tạo mô hình, hàm mất mát và thuật toán tối ưu
    model = get_skin_cancer_model(model_name="resnet50", num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # 5. Vòng lặp huấn luyện chính (Training Loop)
    best_val_acc = 0.0
    os.makedirs("saved_models", exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch [{epoch+1}/{num_epochs}] ---")
        
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
        
        # Lưu lại checkpoint tốt nhất dựa trên độ chính xác tập validation
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            model_path = "saved_models/best_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"--> Saved new best model to {model_path} with Val Acc: {best_val_acc * 100:.2f}%")

if __name__ == "__main__":
    train_model()