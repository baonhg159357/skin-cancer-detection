# skin-cancer-detection
# 🩺 Skin Cancer Detection System (HAM10000)

Hệ thống phân loại và chẩn đoán tổn thương da tự động sử dụng Deep Learning (ResNet50) và kỹ thuật xử lý dữ liệu mất cân bằng nâng cao trên tập dữ liệu chuẩn quốc tế **HAM10000**. Dự án được xây dựng theo chuẩn quy trình MLOps thực chiến, từ tiền xử lý ảnh y tế chuyên sâu, tối ưu hóa hàm mất mát cho đến đóng gói thành Web API phục vụ triển khai thực tế.

---

## 📌 Tổng quan dự án (Project Overview)
Bài toán phân loại 7 nhóm bệnh lý da liễu từ ảnh chụp phóng đại (Dermoscopy) đối mặt với thách thức lớn về sự mất cân bằng dữ liệu trầm trọng giữa các lớp đa số (như nốt ruồi lành tính `nv`) và các lớp thiểu số nguy hiểm (như u sắc tố ác tính `mel`, ung thư tế bào đáy `bcc`,...). 

Dự án áp dụng chiến lược cải tiến toàn diện:
1. **Tiền xử lý ảnh chuyên sâu:** Cắt vùng ROI tập trung tổn thương, khử lông bằng thuật toán DullRazor và cân bằng tương phản cục bộ CLAHE trên không gian màu LAB.
2. **Xử lý mất cân bằng dữ liệu:** Ứng dụng **Smoothed Class Weights** kết hợp **Focal Loss** ($\gamma = 2.0$) để chuyển dịch trọng tâm học tập của mô hình sang các ca khó và lớp thiểu số.
3. **Đóng gói Inference API:** Xây dựng dịch vụ Web API sử dụng **FastAPI** để phục vụ chẩn đoán thời gian thực.

---

## 📂 Cấu trúc mã nguồn (Project Structure)
```text
skin-cancer-detection/
│
├── data/
│   ├── raw/                  # Dữ liệu gốc (HAM10000 metadata & ảnh nguyên bản)
│   └── processed/            # Dữ liệu đã phân chia (train.csv, val.csv, test.csv)
│
├── notebooks/                # Các Jupyter Notebooks thí nghiệm khám phá
│   └── explore_preprocessing.ipynb
│
├── saved_models/             # Lưu trữ trọng số mô hình sau khi huấn luyện
│   └── best_model.pth
│
├── src/                      # Mã nguồn chính của pipeline
│   ├── __init__.py
│   ├── app.py                # FastAPI Inference Service
│   ├── dataset.py            # PyTorch Dataset & Albumentations Transforms
│   ├── evaluate.py           # Script đánh giá chuyên sâu (Accuracy, Recall, F1, Confusion Matrix)
│   ├── loss.py               # Custom Focal Loss implementation
│   ├── model.py              # Kiến trúc ResNet50 Transfer Learning
│   ├── preprocess.py         # Pipeline tiền xử lý (Crop ROI, DullRazor, CLAHE, GroupSplit)
│   └── train.py              # Script huấn luyện mô hình với GPU Acceleration
│
├── config.py                 # Cấu hình hằng số & đường dẫn tập trung
├── .gitignore
└── README.md