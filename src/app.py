import io
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

# Thêm thư mục gốc vào hệ thống path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.model import get_skin_cancer_model
from src.preprocess import HAM10000Preprocessor

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Skin Cancer Detection API",
    description="API chẩn đoán ung thư da (HAM10000) sử dụng ResNet50 và PyTorch",
    version="1.0.0",
)

# Định nghĩa từ điển 7 lớp bệnh HAM10000
CLASSES_DICT = {
    0: "nv (Melanocytic nevi - Nốt ruồi lành tính)",
    1: "mel (Melanoma - Ung thư sắc tố ác tính)",
    2: "bkl (Benign keratosis - Sừng hóa lành tính)",
    3: "bcc (Basal cell carcinoma - Ung thư biểu mô tế bào đáy)",
    4: "akiec (Actinic keratoses / Intraepithelial carcinoma)",
    5: "vasc (Vascular lesions - Tổn thương mạch máu)",
    6: "df (Dermatofibroma - U xơ da)",
}

# Biến toàn cục lưu trữ mô hình và thiết bị
device = None
model = None


@app.on_event("startup")
def load_model():
  global device, model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Loading model on device: {device}")

  # Khởi tạo kiến trúc mô hình (không cần pretrained weights vì sẽ load file fine-tuned)
  model = get_skin_cancer_model(
      model_name="resnet50", num_classes=7, pretrained=False
  )

  # Đường dẫn tới file weights tốt nhất
  model_path = ROOT_DIR / "saved_models" / "best_model.pth"
  if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Không tìm thấy file trọng số mô hình tại: {model_path}"
    )

  # Load weights vào mô hình
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.to(device)
  model.eval()
  print("Đã load thành công mô hình vào FastAPI sẵn sàng phục vụ!")


def preprocess_uploaded_image(image_bytes: bytes) -> np.ndarray:
  """Thực hiện tiền xử lý chuẩn: Crop ROI -> Khử lông DullRazor -> CLAHE"""
  # Chuyển bytes thành mảng numpy OpenCV (BGR)
  nparr = np.frombuffer(image_bytes, np.uint8)
  image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  if image is None:
    raise ValueError("Không thể giải mã file ảnh tải lên.")

  # 1. Cắt ROI
  cropped = HAM10000Preprocessor.crop_lesion_roi(image)
  # 2. Khử lông
  cleaned = HAM10000Preprocessor.remove_hair(cropped)
  # 3. CLAHE
  processed = HAM10000Preprocessor.apply_clahe(cleaned)

  return processed


@app.get("/")
def index():
  return {
      "status": "Running",
      "message": "Skin Cancer Detection API đang hoạt động bình thường!",
  }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
    raise HTTPException(
        status_code=400, detail="Chỉ hỗ trợ định dạng ảnh .jpg, .jpeg, .png"
    )

  try:
    image_bytes = await file.read()
    # Tiền xử lý ảnh giống pipeline huấn luyện
    processed_bgr = preprocess_uploaded_image(image_bytes)

    # Chuyển BGR sang RGB và sang PIL Image để áp dụng transform chuẩn của Val/Test
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(processed_rgb)

    # Định nghĩa transform chuẩn (Resize 224x224 và Normalize ImageNet)
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = inference_transform(pil_img).unsqueeze(0).to(device)

    # Dự đoán
    with torch.no_grad():
      outputs = model(input_tensor)
      probabilities = F.softmax(outputs, dim=1)[0]

    # Lấy kết quả cao nhất
    conf, predicted_idx = torch.max(probabilities, 0)
    predicted_class = CLASSES_DICT[predicted_idx.item()]

    # Trả về top dự đoán và xác suất toàn bộ các lớp
    all_preds = {
        CLASSES_DICT[i]: round(probabilities[i].item() * 100, 2)
        for i in range(7)
    }

    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence_percentage": round(conf.item() * 100, 2),
        "all_class_probabilities": all_preds,
    }

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))