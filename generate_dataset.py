# -*- coding: utf-8 -*-
"""
generate_dataset.py
Sinh dữ liệu ảnh ký tự TỔNG HỢP (synthetic) để huấn luyện SVM khi chưa có
dữ liệu ảnh biển số thật. Mỗi ký tự được vẽ bằng nhiều font OpenCV, nhiều
kích thước, xoay nhẹ, thêm nhiễu/mờ để mô phỏng điều kiện chụp thực tế.

Nếu bạn ĐÃ có dữ liệu ảnh ký tự thật, bỏ qua script này và tổ chức thư mục
dataset/ theo cấu trúc:
    dataset/
        0/  *.png
        1/  *.png
        ...
        A/  *.png
        ...
rồi chạy thẳng train_svm.py.

Cách chạy:
    python generate_dataset.py --out dataset --per_class 250
"""

import os
import argparse
import cv2
import numpy as np

from common import CHARACTERS

FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]


def random_rotate(img, max_angle=10):
    h, w = img.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=0)


def add_noise(img, level=8):
    noise = np.random.normal(0, level, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def random_blur(img):
    if np.random.rand() < 0.5:
        k = np.random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    return img


# def make_char_image(ch, canvas_size=(64, 84)):
#     """Vẽ 1 ký tự lên canvas đen, chữ trắng, với font/độ dày/tỉ lệ ngẫu nhiên."""
#     w, h = canvas_size
#     canvas = np.zeros((h, w), dtype=np.uint8)

#     font = FONTS[np.random.randint(len(FONTS))]
#     thickness = np.random.choice([2, 3, 4])
#     scale = np.random.uniform(1.6, 2.4)

#     (text_w, text_h), baseline = cv2.getTextSize(ch, font, scale, thickness)
#     x = max(0, (w - text_w) // 2 + np.random.randint(-3, 4))
#     y = max(text_h, (h + text_h) // 2 + np.random.randint(-3, 4))

#     cv2.putText(canvas, ch, (x, y), font, scale, 255, thickness, cv2.LINE_AA)

#     canvas = random_rotate(canvas, max_angle=8)
#     canvas = add_noise(canvas, level=10)
#     canvas = random_blur(canvas)
#     return canvas
def make_char_image(ch, canvas_size=(64, 84)):
    """Vẽ 1 ký tự lên canvas đen, chữ trắng, với font/độ dày/tỉ lệ ngẫu nhiên."""
    w, h = canvas_size
    canvas = np.zeros((h, w), dtype=np.uint8)

    font = FONTS[np.random.randint(len(FONTS))]
    thickness = np.random.choice([2, 3, 4])
    scale = np.random.uniform(1.6, 2.4)

    (text_w, text_h), baseline = cv2.getTextSize(ch, font, scale, thickness)
    x = max(0, (w - text_w) // 2 + np.random.randint(-3, 4))
    y = max(text_h, (h + text_h) // 2 + np.random.randint(-3, 4))

    cv2.putText(canvas, ch, (x, y), font, scale, 255, thickness, cv2.LINE_AA)

    canvas = random_rotate(canvas, max_angle=8)

    # --- BƯỚC BỔ SUNG: Cắt sát ký tự (Tight Crop) để giống với ảnh thực tế ---
    # Tìm tọa độ các pixel có màu trắng (pixel của chữ)
    coords = cv2.findNonZero(canvas)
    if coords is not None:
        x_box, y_box, w_box, h_box = cv2.boundingRect(coords)
        pad = 2 # Padding nhẹ giống file plate_detect.py
        x0 = max(0, x_box - pad)
        y0 = max(0, y_box - pad)
        x1 = min(canvas.shape[1], x_box + w_box + pad)
        y1 = min(canvas.shape[0], y_box + h_box + pad)
        # Cắt canvas sát vào chữ
        canvas = canvas[y0:y1, x0:x1]
    # -------------------------------------------------------------------------

    # Thêm nhiễu và làm mờ sau khi đã cắt viền
    canvas = add_noise(canvas, level=10)
    canvas = random_blur(canvas)
    
    return canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dataset", help="Thư mục lưu dataset")
    parser.add_argument("--per_class", type=int, default=250,
                         help="Số ảnh sinh ra cho mỗi ký tự")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for ch in CHARACTERS:
        char_dir = os.path.join(args.out, ch)
        os.makedirs(char_dir, exist_ok=True)
        for i in range(args.per_class):
            img = make_char_image(ch)
            cv2.imwrite(os.path.join(char_dir, f"{ch}_{i:04d}.png"), img)
        print(f"Đã sinh {args.per_class} ảnh cho ký tự '{ch}'")

    print(f"\nHoàn tất. Dataset lưu tại: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
