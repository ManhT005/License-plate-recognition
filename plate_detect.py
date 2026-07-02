# -*- coding: utf-8 -*-
"""
plate_detect.py
Phát hiện vùng biển số xe và tách từng ký tự bằng các kỹ thuật xử lý ảnh
cổ điển của OpenCV (cạnh, contour, tỉ lệ khung hình). SVM chỉ đảm nhận việc
PHÂN LOẠI ký tự sau khi đã tách được, vì SVM không phù hợp để phát hiện
vật thể trên toàn ảnh như CNN/YOLO.
"""

import cv2
import numpy as np


# def _find_plate_candidates(gray, edged, img_area):
#     """Tìm danh sách ứng viên (x, y, w, h) từ ảnh biên (edge) đã cho."""
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]

#     candidates = []
#     for c in contours:
#         x, y, w, h = cv2.boundingRect(c)
#         if h == 0:
#             continue
#         aspect_ratio = w / float(h)
#         area_ratio = (w * h) / float(img_area)
        
       

#         ratio = h / float(w)

#         # lọc nhiễu
#         if h < 25:
#             continue

#         if w < 8:
#             continue

#         if ratio < 1.0 or ratio > 5.5:
#             continue
#         # Biển số VN 1 dòng: tỉ lệ ~2.5-5.5 (dẹt, dài)
#         # Biển số VN 2 dòng: tỉ lệ ~0.8-1.4 (gần vuông)
#         # => chấp nhận khoảng rộng 0.8 - 6.0 để phủ cả 2 loại
#         if 0.8 <= aspect_ratio <= 6.0 and 0.01 <= area_ratio <= 0.85:
#             candidates.append((x, y, w, h, area_ratio))
#     return candidates

def _find_plate_candidates(gray, edged, img_area):
    """Tìm danh sách ứng viên (x, y, w, h) từ ảnh biên (edge) đã cho."""
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Bỏ qua nếu kích thước bằng 0 để tránh lỗi ZeroDivisionError
        if h == 0 or w == 0:
            continue
            
        aspect_ratio = w / float(h)
        area_ratio = (w * h) / float(img_area)

        # Lọc nhiễu kích thước quá nhỏ
        if h < 25 or w < 8:
            continue

        # Biển số VN 1 dòng: tỉ lệ ~2.5-5.5 (dẹt, dài)
        # Biển số VN 2 dòng: tỉ lệ ~0.8-1.4 (gần vuông)
        # => chấp nhận khoảng rộng 0.8 - 6.0 để phủ cả 2 loại
        if 0.8 <= aspect_ratio <= 6.0 and 0.01 <= area_ratio <= 0.85:
            candidates.append((x, y, w, h, area_ratio))
            
    return candidates
def detect_plate(image, debug=False):
    """
    Tìm vùng có khả năng là biển số xe trong ảnh gốc.
    Hỗ trợ cả biển số 1 dòng (dẹt, dài) và 2 dòng (gần vuông).
    image: ảnh BGR
    return: (plate_img, (x, y, w, h)) hoặc (None, None) nếu không tìm thấy
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, 11, 17, 17)
    img_area = image.shape[0] * image.shape[1]

    # --- Phương án 1: phát hiện cạnh bằng Canny ---
    edged = cv2.Canny(gray_blur, 30, 200)
    candidates = _find_plate_candidates(gray_blur, edged, img_area)

    # --- Phương án 2 (dự phòng): nhị phân hoá Otsu + morphology ---
    if not candidates:
        _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        edged2 = cv2.Canny(closed, 30, 200)
        candidates = _find_plate_candidates(gray_blur, edged2, img_area)

    if not candidates:
        return None, None

    # Ưu tiên vùng có diện tích lớn nhất trong số các ứng viên hợp lệ
    best_box = max(candidates, key=lambda c: c[4])
    x, y, w, h, _ = best_box

    pad = int(0.02 * w)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    plate_img = image[y0:y1, x0:x1]
    return plate_img, (x0, y0, x1 - x0, y1 - y0)


def _find_char_candidates(gray, thresh):
    H, W = gray.shape[:2]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        height_ratio = h / float(H)
        width_ratio = w / float(W)
        aspect_ratio = w / float(h) if h > 0 else 0

        # Loại các contour quá to gần bằng cả khung (khả năng là viền/khung biển,
        # không phải ký tự), và các contour quá nhỏ (nhiễu, chấm, vết bẩn)
        if width_ratio > 0.92 or height_ratio > 0.98:
            continue
        if 0.10 <= height_ratio <= 0.95 and 0.08 <= aspect_ratio <= 1.3 and w > 3 and h > 8:
            candidates.append((x, y, w, h))
    return candidates


def segment_characters(plate_img):
    """
    Tách các ký tự riêng lẻ từ ảnh vùng biển số đã cắt.
    Hỗ trợ cả biển số 1 dòng và 2 dòng (sắp xếp theo hàng rồi theo cột).
    return: list các tuple (char_img, (x, y, w, h)) đã sắp xếp trái -> phải,
            toạ độ (x, y) tính theo hệ toạ độ của plate_img gốc (chưa cắt viền)
    """
    H0, W0 = plate_img.shape[:2]

    # Cắt bớt viền ngoài của biển số (thường có khung/viền đen liền khối,
    # nếu không loại bỏ sẽ khiến bước tìm contour dính liền thành 1 khối)
    margin_x = max(2, int(0.05 * W0))
    margin_y = max(2, int(0.05 * H0))
    x_off, y_off = margin_x, margin_y
    inner = plate_img[margin_y:H0 - margin_y, margin_x:W0 - margin_x]
    if inner.size == 0:
        inner = plate_img
        x_off, y_off = 0, 0

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Phương án 1: Otsu ---
    _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates = _find_char_candidates(gray, thresh)

    # --- Phương án 2 (dự phòng): adaptive threshold, tốt hơn khi ánh sáng không đều ---
    if not candidates:
        thresh2 = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 10
        )
        candidates = _find_char_candidates(gray, thresh2)

    if not candidates:
        return []

    H, W = gray.shape[:2]
    # Sắp xếp theo hàng (y) rồi theo cột (x) -- hỗ trợ biển số 2 dòng
    candidates.sort(key=lambda b: (b[1] // (H // 2 + 1), b[0]))

    chars = []
    for (x, y, w, h) in candidates:
        pad = 2
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        char_img = gray[y0:y1, x0:x1]
        # trả về toạ độ theo hệ quy chiếu của plate_img gốc (cộng lại phần đã cắt viền)
        chars.append((char_img, (x0 + x_off, y0 + y_off, x1 - x0, y1 - y0)))

    return chars