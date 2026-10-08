import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss cho bài toán phân loại đa lớp (Multi-class Classification).
        - alpha: Trọng số cho từng lớp (có thể dùng lại Smoothed Class Weights)
        - gamma: Hệ số tập trung vào mẫu khó (gamma càng lớn, mô hình càng bỏ qua mẫu dễ)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # tensor trọng số các lớp (nếu có)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [Batch_size, Num_classes] (logits thô chưa qua softmax)
        # targets: [Batch_size] (nhãn dạng số nguyên)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss) # Xác suất dự đoán đúng của mô hình cho nhãn thực tế
        
        # Tính toán thành phần Focal Loss
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            # Nhân thêm trọng số alpha tương ứng với từng lớp của mẫu
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss