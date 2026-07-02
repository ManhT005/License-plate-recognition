# Nhận dạng biển số xe bằng SVM 

Dự án gồm pipeline hoàn chỉnh: **phát hiện vùng biển số → tách ký tự → nhận
dạng ký tự bằng SVM**, kèm giao diện Tkinter để test trực quan.

## Kiến trúc pipeline

1. **Phát hiện biển số** (`plate_detect.py::detect_plate`): dùng lọc cạnh
   Canny + tìm contour có tỉ lệ khung hình giống biển số (OpenCV cổ điển,
   không cần huấn luyện).
2. **Tách ký tự** (`plate_detect.py::segment_characters`): nhị phân hoá
   (Otsu) vùng biển số rồi tìm contour từng ký tự, lọc theo tỉ lệ kích
   thước, sắp xếp trái→phải (hỗ trợ biển 2 dòng).
3. **Trích đặc trưng HOG** (`common.py`): mỗi ảnh ký tự sau khi chuẩn hoá
   kích thước 24x32 được biến thành vector đặc trưng HOG.
4. **Phân loại bằng SVM** (`train_svm.py`, `sklearn.svm.SVC`, kernel RBF):
   nhận vector HOG, dự đoán ký tự (0-9, A-Y trừ các chữ dễ nhầm lẫn
   I, O, Q, R, W, Z, J).
5. **Giao diện** (`gui_app.py`): Tkinter, cho phép chọn ảnh, chạy toàn bộ
   pipeline, hiển thị khung nhận dạng + chuỗi biển số.

> **Vì sao dùng SVM chỉ cho bước phân loại ký tự, không phải phát hiện
> biển số?** SVM là bộ phân loại (classification), không phải bộ phát
> hiện vật thể (detection) như CNN/YOLO. Cách kết hợp OpenCV (detection cổ
> điển) + SVM (classification) là mô hình truyền thống phổ biến trước khi
> deep learning (YOLO, CNN OCR) trở nên phổ biến.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Các bước chạy

### 1. Chuẩn bị dữ liệu huấn luyện ký tự

Nếu **chưa có** dữ liệu ảnh ký tự thật, sinh dữ liệu tổng hợp:

```bash
python generate_dataset.py --out dataset --per_class 250
```

Nếu **đã có** ảnh ký tự thật (khuyến nghị để độ chính xác cao hơn), tổ
chức theo cấu trúc:

```
dataset/
    0/   *.png hoặc *.jpg (ảnh ký tự "0")
    1/   ...
    A/   ...
    ...
```

### 2. Huấn luyện SVM

```bash
python train_svm.py --data dataset --model_out model
```

Thêm `--grid_search` nếu muốn tự động dò tham số C/gamma tốt nhất (chậm
hơn nhưng chính xác hơn):

```bash
python train_svm.py --data dataset --model_out model --grid_search
```

Sau khi chạy xong, mô hình được lưu tại `model/svm_model.pkl` và
`model/scaler.pkl`.

### 3. Chạy giao diện

```bash
python gui_app.py
```

- Nhấn **"Chọn ảnh"** để nạp ảnh có chứa xe/biển số.
- Nhấn **"Nhận dạng"** để chạy pipeline: ảnh hiển thị khung xanh (vùng
  biển số) và khung xanh dương (từng ký tự), kết quả biển số hiện ở dưới.

## Lưu ý về độ chính xác

- Dữ liệu **tổng hợp** (`generate_dataset.py`) giúp mô hình chạy được ngay
  nhưng độ chính xác trên ảnh thực tế sẽ **thấp hơn** nhiều so với việc
  huấn luyện trên ảnh ký tự cắt từ biển số thật.
- Bước `detect_plate` dùng kỹ thuật cổ điển (contour + tỉ lệ khung hình)
  nên nhạy với ánh sáng/góc chụp; với ảnh chụp thẳng, nền rõ, biển số nổi
  bật thì hoạt động tốt nhất.
- Nếu bạn cần độ chính xác cao và ổn định hơn cho báo cáo/đồ án, hướng
  YOLOv8 (detection) + CNN OCR sẽ cho kết quả tốt hơn nhiều so với SVM,
  vì SVM không tự học đặc trưng mà phụ thuộc vào HOG (đặc trưng thủ công).

## Cấu trúc file

```
license_plate_svm/
├── common.py              # tiền xử lý ảnh + trích đặc trưng HOG
├── generate_dataset.py    # sinh dữ liệu ký tự tổng hợp
├── train_svm.py           # huấn luyện SVM
├── plate_detect.py        # phát hiện biển số + tách ký tự (OpenCV)
├── gui_app.py             # giao diện Tkinter
├── requirements.txt
└── README.md
```
