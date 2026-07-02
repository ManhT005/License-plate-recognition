# -*- coding: utf-8 -*-
"""
common.py
Các hàm & hằng số dùng chung cho toàn bộ pipeline:
 - Danh sách ký tự biển số xe Việt Nam (không dùng I, O, Q, R, W, Z, J để tránh nhầm lẫn)
 - Tiền xử lý ảnh ký tự (grayscale, resize, nhị phân hoá)
 - Trích đặc trưng HOG (Histogram of Oriented Gradients) làm đầu vào cho SVM
"""

import cv2
import numpy as np
from skimage.feature import hog

# Các ký tự thường xuất hiện trên biển số xe Việt Nam
CHARACTERS = list("0123456789ABCDEFGHKLMNPSTUVXY")

# Kích thước chuẩn hoá ảnh ký tự trước khi trích đặc trưng (rộng, cao)
CHAR_SIZE = (24, 32)

# Tham số HOG - giữ cố định giữa lúc train và lúc predict
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)


# def preprocess_char(img):
#     """
#     Chuẩn hoá 1 ảnh ký tự về dạng nhị phân, kích thước cố định, nền đen chữ trắng.
#     img: ảnh grayscale hoặc BGR (numpy array)
#     return: ảnh nhị phân uint8, kích thước CHAR_SIZE
#     """
#     if img is None:
#         raise ValueError("Ảnh đầu vào rỗng (None)")

#     if len(img.shape) == 3:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img

#     # Nhị phân hoá bằng Otsu, tự động đảo màu nếu nền sáng
#     # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     binary = cv2.adaptiveThreshold(
#     gray,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY_INV,
#     31,
#     15
# )
#     if np.mean(binary) > 127:
#         binary = cv2.bitwise_not(binary)

#     resized = cv2.resize(binary, CHAR_SIZE, interpolation=cv2.INTER_AREA)
#     return resized

def preprocess_char(img):
    """
    Chuẩn hoá 1 ảnh ký tự về dạng nhị phân, kích thước cố định, nền đen chữ trắng.
    img: ảnh grayscale hoặc BGR (numpy array)
    return: ảnh nhị phân uint8, kích thước CHAR_SIZE
    """
    if img is None:
        raise ValueError("Ảnh đầu vào rỗng (None)")

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Dùng Otsu threshold thay vì Adaptive. Otsu cực kỳ ổn định cho ảnh ký tự đã cắt nhỏ.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Đảm bảo đầu ra luôn là chữ trắng nền đen.
    # Nếu phần lớn diện tích là màu trắng (> 127) => nền đang trắng => đảo ngược (bitwise_not)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    resized = cv2.resize(binary, CHAR_SIZE, interpolation=cv2.INTER_AREA)
    return resized
def extract_hog(img):
    """
    Trích đặc trưng HOG từ ảnh ký tự đã tiền xử lý (hoặc ảnh thô, sẽ tự preprocess).
    return: vector đặc trưng 1 chiều (numpy array)
    """
    if img.shape[:2] != (CHAR_SIZE[1], CHAR_SIZE[0]):
        img = preprocess_char(img)
    features = hog(img, **HOG_PARAMS)
    return features


def char_to_label(ch):
    return CHARACTERS.index(ch)


def label_to_char(label):
    return CHARACTERS[int(label)]
