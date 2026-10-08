import os
import torch
import torch.nn as nn
from torchvision import models

def get_skin_cancer_model(
    model_name: str = "resnet50", 
    num_classes: int = 7, 
    pretrained: bool = True, 
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Khởi tạo mô hình Deep Learning cho bài toán phân loại ung thư da.
    Hỗ trợ ResNet50 hoặc EfficientNet-B2, kèm tùy chọn đóng băng phần trích xuất đặc trưng.
    """
    # Chuyển hướng cache sang ổ D để tránh tràn ổ C
    custom_cache_dir = "D:/torch_cache"
    os.makedirs(custom_cache_dir, exist_ok=True)
    torch.hub.set_dir(custom_cache_dir)
    os.environ['TORCH_HOME'] = custom_cache_dir

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        
        # Đóng băng các tầng backbone nếu cần để ổn định huấn luyện ban đầu
        if pretrained and freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # Thay thế lớp Fully Connected cuối cùng
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),  # Tăng nhẹ dropout để chống overfitting mạnh hơn
            nn.Linear(in_features, num_classes)
        )
        
        # Nếu đóng băng backbone, mở khóa riêng cho lớp fc vừa tạo
        if pretrained and freeze_backbone:
            for param in model.fc.parameters():
                param.requires_grad = True

    elif model_name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        
        if pretrained and freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
        
        if pretrained and freeze_backbone:
            for param in model.classifier.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Model architecture '{model_name}' không được hỗ trợ!")

    return model

if __name__ == "__main__":
    model = get_skin_cancer_model(model_name="resnet50", num_classes=7, freeze_backbone=True)
    print("Khởi tạo mô hình thành công với tùy chọn Freeze Backbone!")