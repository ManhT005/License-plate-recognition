# Nhận diện Biển số xe (License Plate Recognition - LPR)

Hệ thống nhận diện biển số xe máy và ô tô tại Việt Nam sử dụng mô hình học sâu (Deep Learning). Hệ thống tích hợp mô hình phát hiện biển số nghiêng (Oriented Bounding Box - OBB) và nhận diện ký tự (OCR) bằng YOLOv8 hoặc mạng CNN tự thiết kế (CharCNN).

---

## 🚀 Tính năng nổi bật

- **YOLOv8 OBB Detection**: Định vị chính xác góc xoay và tọa độ của biển số xe, giúp cắt (crop) và căn chỉnh thẳng biển số bị nghiêng một cách hoàn hảo.
- **Hai cơ chế OCR linh hoạt**:
  - **YOLOv8 OCR**: Nhận diện trực tiếp ký tự trên biển số.
  - **Custom CharCNN (PyTorch)**: Tự động phân ngưỡng ảnh (Thresholding), tách từng ký tự bằng đường bao (Contours) và phân loại qua mạng CNN tự huấn luyện.
- **Giao diện Web Demo (Flask)**:
  - Tải ảnh lên và xem kết quả trực quan.
  - Chế độ **Debug Segmentation** để xem quá trình tiền xử lý và cắt từng ký tự.
  - Chế độ **Debug OBB Points** hiển thị tọa độ các đỉnh của biển số được phát hiện.

---

## 📁 Cấu trúc thư mục

```text
├── demo/
│   ├── models/                   # Thư mục chứa các file weights huấn luyện (.pt, .pth)
│   ├── static/                   # Ảnh tạm thời lưu trong quá trình chạy web
│   ├── templates/                # Giao diện HTML của trang chủ
│   ├── main.py                   # File chính chạy ứng dụng Flask
│   ├── pipeline.py               # Quy trình xử lý dùng mô hình YOLOv8 OCR
│   └── diy_charCNN_pipeline.py   # Quy trình xử lý dùng thuật toán tách ký tự + CharCNN
├── src/
│   ├── train_detection.py        # Kịch bản huấn luyện mô hình YOLOv8 OBB phát hiện biển số
│   ├── train_ocr.py              # Định nghĩa mạng CharCNN bằng PyTorch
│   ├── utils.py                  # Các hàm bổ trợ
│   └── inference.py              # Script chạy thử nghiệm offline
├── notebooks/                    # Jupyter Notebooks huấn luyện trên Kaggle/Colab
│   ├── kaggle_detection_train.ipynb
│   ├── kaggle_yolo_ocr.ipynb
│   └── kaggle_char_CNN.ipynb
├── requirements.txt              # Danh sách các thư viện phụ thuộc
└── README.md                     # Tài liệu hướng dẫn sử dụng
```

---

## 🛠️ Hướng dẫn cài đặt

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python (phiên bản khuyên dùng `>= 3.9`). Tiến hành cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị Weights (Trọng số mô hình)
Trước khi chạy ứng dụng Flask, bạn cần đặt các file weights vào thư mục `demo/models/`:
- Mô hình phát hiện biển số (`license_detection.pt`)
- Mô hình nhận diện ký tự (`yolo_ocr.pt` hoặc mô hình CharCNN `best_char_cnn.pth`)

---

## 💻 Hướng dẫn chạy thử nghiệm

### Chạy ứng dụng Web Demo (Flask App)
Khởi chạy ứng dụng Flask để trải nghiệm giao diện người dùng:

```bash
python demo/main.py
```

Sau khi chạy lệnh, truy cập địa chỉ **`http://localhost:5000`** trên trình duyệt web để tải ảnh lên và kiểm tra kết quả nhận diện.

---

## 🏋️ Huấn luyện mô hình (Training)

### 1. Huấn luyện mô hình Phát hiện Biển số (YOLO OBB)
Chuẩn bị file cấu hình `data.yaml` và chạy file training:

```bash
python src/train_detection.py
```

### 2. Huấn luyện mô hình Nhận diện ký tự (OCR)
Bạn có thể tham khảo cấu trúc mạng trong `src/train_ocr.py` hoặc sử dụng các file notebook mẫu trong thư mục `notebooks/` để chạy trực tiếp trên Kaggle hoặc Google Colab với tài nguyên GPU miễn phí.
