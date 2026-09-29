import os
import torch
import torch.nn as nn
from torchvision import models

def get_skin_cancer_model(model_name: str = "resnet50", num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """
    Khởi tạo mô hình Deep Learning cho bài toán phân loại ung thư da.
    Hỗ trợ các kiến trúc phổ biến như ResNet50 hoặc EfficientNet-B2.
    """
    # CHUYỂN HƯỚNG CACHE SANG Ổ D ĐỂ TRÁNH LỖI TRÀN Ổ C
    custom_cache_dir = "D:/torch_cache"
    os.makedirs(custom_cache_dir, exist_ok=True)
    torch.hub.set_dir(custom_cache_dir)
    os.environ['TORCH_HOME'] = custom_cache_dir

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        
    elif model_name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Model architecture '{model_name}' is not supported yet.")

    return model

if __name__ == "__main__":
    model = get_skin_cancer_model(model_name="resnet50", num_classes=7)
    print("Khởi tạo mô hình thành công!")