# Sử dụng base image Python chính thức (phiên bản slim gọn nhẹ)
FROM python:3.10-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết cho OpenCV và xử lý ảnh
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements vào container và cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn dự án (src, config.py, saved_models,...) vào container
COPY . .

# Mở cổng 8000 để truy cập ứng dụng FastAPI từ bên ngoài
EXPOSE 8000

# Lệnh mặc định khi container khởi chạy (chạy FastAPI bằng Uvicorn)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]