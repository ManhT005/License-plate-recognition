from ultralytics import YOLO

def train():
    # Dùng YOLOv8s — nhỏ hơn medium nhưng đủ tốt cho 4GB VRAM
    model = YOLO('yolov8s.pt')  # tự download pretrained weights

    results = model.train(
        data='data.yaml',        # đường dẫn từ thư mục gốc project
        epochs=50,
        imgsz=640,
        batch=8,                 # 4GB VRAM dùng batch=8
        device=0,                # GPU, đổi thành 'cpu' nếu CUDA lỗi
        project='runs',
        name='license_plate_v1',
        patience=10,             # early stopping
        save=True,
        plots=True               # tự vẽ confusion matrix, PR curve
    )

    print("Train xong!")
    print("Weights lưu tại: runs/license_plate_v1/weights/best.pt")

if __name__ == '__main__':
    train()