# -*- coding: utf-8 -*-
"""
train_svm.py
Huấn luyện mô hình SVM nhận dạng ký tự biển số xe từ dataset ảnh ký tự.

Cấu trúc dataset yêu cầu:
    dataset/
        0/ *.png hoặc *.jpg
        1/ ...
        A/ ...
        ...

Cách chạy:
    python train_svm.py --data dataset --model_out model
"""

import os
import glob
import argparse
import numpy as np
import cv2
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from common import CHARACTERS, preprocess_char, extract_hog, char_to_label


def load_dataset(data_dir):
    X, y = [], []
    for ch in CHARACTERS:
        char_dir = os.path.join(data_dir, ch)
        if not os.path.isdir(char_dir):
            print(f"[Cảnh báo] Không tìm thấy thư mục cho ký tự '{ch}', bỏ qua.")
            continue
        paths = glob.glob(os.path.join(char_dir, "*"))
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            proc = preprocess_char(img)
            feat = extract_hog(proc)
            X.append(feat)
            y.append(char_to_label(ch))
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset", help="Thư mục dataset")
    parser.add_argument("--model_out", default="model", help="Thư mục lưu model")
    parser.add_argument("--grid_search", action="store_true",
                         help="Bật GridSearchCV để dò tham số C, gamma (chậm hơn)")
    args = parser.parse_args()

    print("Đang nạp dữ liệu và trích đặc trưng HOG...")
    X, y = load_dataset(args.data)
    print(f"Tổng số mẫu: {len(X)}, số lớp: {len(set(y))}")

    if len(X) == 0:
        print("Không có dữ liệu để huấn luyện. Hãy chạy generate_dataset.py trước "
              "hoặc chuẩn bị dataset thật theo đúng cấu trúc thư mục.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if args.grid_search:
        print("Đang chạy GridSearchCV (có thể mất vài phút)...")
        param_grid = {
            "C": [1, 5, 10, 50],
            "gamma": ["scale", 0.01, 0.001],
            "kernel": ["rbf"],
        }
        grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1, verbose=1)
        grid.fit(X_train_s, y_train)
        clf = grid.best_estimator_
        print("Tham số tốt nhất:", grid.best_params_)
    else:
        clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nĐộ chính xác trên tập test: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=CHARACTERS,
                                 labels=list(range(len(CHARACTERS))), zero_division=0))

    os.makedirs(args.model_out, exist_ok=True)
    joblib.dump(clf, os.path.join(args.model_out, "svm_model.pkl"))
    joblib.dump(scaler, os.path.join(args.model_out, "scaler.pkl"))
    print(f"\nĐã lưu mô hình vào thư mục: {os.path.abspath(args.model_out)}")


if __name__ == "__main__":
    main()
