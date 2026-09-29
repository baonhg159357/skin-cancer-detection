import torch

def check_device() -> None:
    """
    Hàm kiểm tra xem PyTorch đang chạy trên CPU hay GPU NVIDIA 
    và hiển thị các thông tin chi tiết về phần cứng.
    """
    # In ra phiên bản PyTorch hiện tại đang được sử dụng trong môi trường
    print(f"PyTorch version: {torch.__version__}")
    
    # Kiểm tra xem CUDA (tính năng tăng tốc phần cứng cho GPU NVIDIA) có sẵn sàng không
    if torch.cuda.is_available():
        print("Status: GPU IS READY!")
        
        # Lấy tên của card đồ họa (ví dụ: NVIDIA GeForce RTX...) tại thiết bị 0
        print(f"- GPU Name: {torch.cuda.get_device_name(0)}")
        
        # Đếm tổng số lượng GPU khả dụng trên hệ thống máy tính
        print(f"- Available GPU Count: {torch.cuda.device_count()}")
    else:
        # Trường hợp không tìm thấy card đồ họa hoặc chưa cài đặt bản PyTorch hỗ trợ CUDA
        print("Status: No compatible GPU found. Running on CPU.")

if __name__ == "__main__":
    # Điểm khởi chạy chính khi thực thi file script qua terminal
    check_device()