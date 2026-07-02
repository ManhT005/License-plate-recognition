# -*- coding: utf-8 -*-
"""
gui_app.py
Giao diện đồ hoạ (Tkinter) cho ứng dụng nhận dạng biển số xe dùng SVM.

Chức năng:
 - Chọn ảnh từ máy tính
 - Tự động phát hiện vùng biển số (OpenCV) và tách ký tự
 - Dùng mô hình SVM (đã huấn luyện ở train_svm.py) để nhận dạng từng ký tự
 - Hiển thị ảnh với khung biển số/ký tự + kết quả biển số dạng text

Cách chạy:
    python gui_app.py
(Yêu cầu đã có model/svm_model.pkl và model/scaler.pkl từ bước train_svm.py)
"""

import os

# Ép biến môi trường locale sang UTF-8 TRƯỚC khi import tkinter, để Tcl/Tk
# khởi tạo với encoding hệ thống đúng ngay từ đầu (phòng trường hợp locale
# hệ điều hành WSL chưa được set về UTF-8, gây lỗi hiển thị tiếng Việt dạng
# \u1eacN...). Chỉ set khi biến chưa tồn tại để không ghi đè cấu hình đã có.
# os.environ.setdefault("LANG", "en_US.UTF-8")
# os.environ.setdefault("LC_ALL", "en_US.UTF-8")
# os.environ.setdefault("LANGUAGE", "en_US.UTF-8")

import io
import urllib.request
import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

from common import extract_hog, label_to_char
from plate_detect import detect_plate, segment_characters

MODEL_DIR = "model"

# Bảng màu giao diện
BG_MAIN = "#f4f6f8"
BG_HEADER = "#1f2a44"
BG_CARD = "#ffffff"
BG_CANVAS = "#e7eaee"
COLOR_TEXT = "#1f2a44"
COLOR_MUTED = "#6b7280"
COLOR_ACCENT = "#2563eb"
COLOR_SUCCESS = "#16a34a"
COLOR_RESULT = "#dc2626"


def pick_font():
    """
    Chọn 1 font có hỗ trợ tiếng Việt có sẵn trên hệ thống.
    Trên Windows: 'Segoe UI'. Trên Linux/WSL: 'DejaVu Sans' hoặc 'Noto Sans'.
    Tránh lỗi hiển thị escape unicode (\\u1eE8NG...) khi font không có dấu.
    """
    available = set(tkfont.families())
    candidates = ["Segoe UI", "Noto Sans", "DejaVu Sans", "Ubuntu", "Arial", "Helvetica"]
    for name in candidates:
        if name in available:
            return name
    return "TkDefaultFont"


class CameraWindow(tk.Toplevel):
    """
    Cửa sổ xem trước hình ảnh trực tiếp từ camera (webcam), cho phép chụp
    1 khung hình để đưa vào pipeline nhận dạng.
    """

    def __init__(self, parent, on_capture, font_family, camera_index=0):
        super().__init__(parent)
        self.title("Chụp ảnh từ Camera")
        self.configure(bg=BG_MAIN)
        self.resizable(False, False)
        self.on_capture = on_capture
        self.font_family = font_family
        self._running = False
        self._last_frame = None

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Không mở được Camera",
                "Không tìm thấy hoặc không truy cập được camera.\n\n"
                "Nếu đang dùng WSL2: camera USB cần được gắn (attach) vào WSL "
                "bằng công cụ usbipd-win trước, WSL2 không tự thấy webcam của "
                "Windows theo mặc định."
            )
            self.destroy()
            return

        self.video_label = tk.Label(self, bg="#000000")
        self.video_label.pack(padx=10, pady=10)

        btn_frame = tk.Frame(self, bg=BG_MAIN)
        btn_frame.pack(pady=(0, 10))

        tk.Button(btn_frame, text="Chụp ảnh", command=self.capture,
                  bg=COLOR_SUCCESS, fg="white", font=(font_family, 11, "bold"),
                  relief="flat", padx=14, pady=6).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Đóng", command=self.close,
                  bg="#9ca3af", fg="white", font=(font_family, 11),
                  relief="flat", padx=14, pady=6).pack(side="left", padx=5)

        self.protocol("WM_DELETE_WINDOW", self.close)
        self._running = True
        self._update_frame()

    def _update_frame(self):
        if not self._running:
            return
        ret, frame = self.cap.read()
        if ret:
            self._last_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((640, 480))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.video_label.configure(image=tk_img)
            self.video_label.image = tk_img
        self.after(30, self._update_frame)

    def capture(self):
        if self._last_frame is not None:
            self.on_capture(self._last_frame.copy())
        self.close()

    def close(self):
        self._running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.destroy()


class PlateRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.font_family = pick_font()
        print("TEST:", "Nhận dạng biển số xe")
        self.root.title("Nhận dạng biển số xe - SVM")
        self.root.geometry("960x680")
        self.root.minsize(820, 600)
        self.root.configure(bg=BG_MAIN)

        self.clf = None
        self.scaler = None
        self.cv_image = None  # ảnh gốc (BGR, numpy)

        self._setup_style()
        self._build_ui()
        self._load_model()

    # -------------------------------------------------------- FONT/STYLE ----
    def f(self, size, weight="normal"):
        return (self.font_family, size, weight)

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Accent.TButton", font=self.f(11, "bold"),
                         background=COLOR_ACCENT, foreground="white",
                         padding=10, borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#1d4ed8"),
                                                  ("disabled", "#93b4f5")])

        style.configure("Success.TButton", font=self.f(11, "bold"),
                         background=COLOR_SUCCESS, foreground="white",
                         padding=10, borderwidth=0)
        style.map("Success.TButton", background=[("active", "#15803d"),
                                                   ("disabled", "#9adcb0")])

    # ---------------------------------------------------------- UI ----
    def _build_ui(self):
        # ---- Thanh tiêu đề ----
        header = tk.Frame(self.root, bg=BG_HEADER, height=68)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="NHẬN DẠNG BIỂN SỐ XE",
                 font=self.f(18, "bold"), bg=BG_HEADER, fg="white").pack(side="left", padx=24)
        tk.Label(header, text="Mô hình SVM  ·  HOG feature",
                 font=self.f(10), bg=BG_HEADER, fg="#b8c2d9").pack(side="left", padx=4)

        # ---- Vùng nội dung chính ----
        body = tk.Frame(self.root, bg=BG_MAIN)
        body.pack(fill="both", expand=True, padx=20, pady=16)

        # Thanh công cụ (nút bấm)
        toolbar = tk.Frame(body, bg=BG_MAIN)
        toolbar.pack(fill="x", pady=(0, 8))

        self.btn_load = ttk.Button(toolbar, text="Chọn ảnh", style="Accent.TButton",
                                    command=self.load_image)
        self.btn_load.pack(side="left")

        self.btn_camera = ttk.Button(toolbar, text="Chụp từ Camera", style="Accent.TButton",
                                      command=self.open_camera)
        self.btn_camera.pack(side="left", padx=10)

        self.btn_predict = ttk.Button(toolbar, text="Nhận dạng", style="Success.TButton",
                                       command=self.predict, state="disabled")
        self.btn_predict.pack(side="left")

        # Thanh nhập URL ảnh trên mạng
        url_bar = tk.Frame(body, bg=BG_MAIN)
        url_bar.pack(fill="x", pady=(0, 12))

        tk.Label(url_bar, text="Link ảnh:", font=self.f(10), bg=BG_MAIN,
                 fg=COLOR_TEXT).pack(side="left", padx=(0, 6))

        self.url_var = tk.StringVar()
        self.url_entry = tk.Entry(url_bar, textvariable=self.url_var, font=self.f(10),
                                   relief="solid", borderwidth=1)
        self.url_entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.url_entry.bind("<Return>", lambda e: self.load_image_from_url())

        self.btn_url = ttk.Button(url_bar, text="Tải từ URL", style="Accent.TButton",
                                   command=self.load_image_from_url)
        self.btn_url.pack(side="left", padx=(8, 0))

        # Khung hiển thị ảnh (card)
        canvas_card = tk.Frame(body, bg=BG_CARD, highlightbackground="#d1d5db",
                                highlightthickness=1)
        canvas_card.pack(fill="both", expand=True)

        self.canvas = tk.Label(canvas_card, bg=BG_CANVAS,
                                text="Chưa có ảnh — nhấn \"Chọn ảnh\" để bắt đầu",
                                font=self.f(11), fg=COLOR_MUTED)
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)

        # Khung kết quả (card)
        result_card = tk.Frame(body, bg=BG_CARD, highlightbackground="#d1d5db",
                                highlightthickness=1)
        result_card.pack(fill="x", pady=(12, 0))

        inner = tk.Frame(result_card, bg=BG_CARD)
        inner.pack(fill="x", padx=18, pady=14)

        tk.Label(inner, text="BIỂN SỐ NHẬN DẠNG ĐƯỢC", font=self.f(10, "bold"),
                 bg=BG_CARD, fg=COLOR_MUTED).pack(anchor="w")

        self.result_var = tk.StringVar(value="—")
        tk.Label(inner, textvariable=self.result_var, font=self.f(28, "bold"),
                 fg=COLOR_RESULT, bg=BG_CARD).pack(anchor="w", pady=(2, 0))

        # ---- Thanh trạng thái ----
        status_bar = tk.Frame(self.root, bg="#e5e7eb", height=30)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)

        self.status_var = tk.StringVar(value="Đang khởi động...")
        tk.Label(status_bar, textvariable=self.status_var, font=self.f(9),
                 bg="#e5e7eb", fg=COLOR_MUTED, anchor="w").pack(fill="both", padx=12)

    # --------------------------------------------------------- MODEL ----
    def _load_model(self):
        model_path = os.path.join(MODEL_DIR, "svm_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.clf = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.status_var.set("Đã nạp mô hình SVM thành công. Hãy chọn ảnh để nhận dạng.")
        else:
            self.status_var.set(
                "Chưa tìm thấy mô hình! Hãy chạy generate_dataset.py rồi train_svm.py trước."
            )
            messagebox.showwarning(
                "Thiếu mô hình",
                "Không tìm thấy model/svm_model.pkl.\n"
                "Vui lòng chạy:\n  python generate_dataset.py\n  python train_svm.py\n"
                "trước khi sử dụng giao diện này."
            )

    # --------------------------------------------------------- ACTIONS ----
    def open_camera(self):
        CameraWindow(self.root, self._on_camera_capture, self.font_family)

    def _on_camera_capture(self, frame):
        self._set_new_image(frame, "Đã chụp ảnh từ camera  ·  nhấn \"Nhận dạng\" để xử lý")

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh này.")
            return
        self._set_new_image(img, f"Đã nạp ảnh: {os.path.basename(path)}  ·  nhấn \"Nhận dạng\" để xử lý")

    def load_image_from_url(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showinfo("Thiếu link", "Vui lòng dán link ảnh (http/https) vào ô trên.")
            return
        if not (url.startswith("http://") or url.startswith("https://")):
            messagebox.showerror("Link không hợp lệ", "Link ảnh phải bắt đầu bằng http:// hoặc https://")
            return

        self.status_var.set("Đang tải ảnh từ URL...")
        self.root.update_idletasks()

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (compatible; PlateRecognizer/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_data = resp.read()

            # Giải mã bytes ảnh (jpg/png/...) thành ảnh OpenCV (BGR)
            file_bytes = np.frombuffer(raw_data, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Dữ liệu tải về không phải ảnh hợp lệ (jpg/png/bmp).")

            self._set_new_image(img, f"Đã tải ảnh từ URL  ·  nhấn \"Nhận dạng\" để xử lý")

        except Exception as e:
            self.status_var.set("Tải ảnh từ URL thất bại.")
            messagebox.showerror(
                "Không tải được ảnh",
                f"Không thể tải ảnh từ link đã nhập.\n\nChi tiết lỗi:\n{e}\n\n"
                "Gợi ý: một số trang web (Facebook, Pinterest, Google Images...) chặn tải "
                "trực tiếp hoặc yêu cầu đăng nhập. Hãy thử click phải vào ảnh trên trình "
                "duyệt → \"Copy image address\" để lấy đúng link ảnh gốc (thường có đuôi "
                ".jpg/.png), hoặc tải ảnh về máy rồi dùng nút \"Chọn ảnh\"."
            )

    def _set_new_image(self, img, status_text):
        self.cv_image = img
        self._show_image(img)
        self.result_var.set("—")
        self.btn_predict.config(state="normal")
        self.status_var.set(status_text)

    def predict(self):
        if self.cv_image is None:
            return
        if self.clf is None or self.scaler is None:
            messagebox.showwarning("Thiếu mô hình", "Chưa có mô hình SVM đã huấn luyện.")
            return

        display_img = self.cv_image.copy()
        plate_img, box = detect_plate(self.cv_image)

        if plate_img is None or plate_img.size == 0:
            self.status_var.set("Không phát hiện được vùng biển số trong ảnh.")
            self.result_var.set("Không tìm thấy")
            self._show_image(display_img)
            return

        x, y, w, h = box
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        chars = segment_characters(plate_img)
        if not chars:
            self.status_var.set("Phát hiện được biển số nhưng không tách được ký tự.")
            self.result_var.set("—")
            self._show_image(display_img)
            return

        plate_text = ""
        for char_img, (cx, cy, cw, ch) in chars:
            feat = extract_hog(char_img).reshape(1, -1)
            feat_s = self.scaler.transform(feat)
            pred = self.clf.predict(feat_s)[0]
            ch_label = label_to_char(pred)
            plate_text += ch_label

            # vẽ khung ký tự lên ảnh gốc (toạ độ tương đối theo vùng plate)
            cv2.rectangle(display_img, (x + cx, y + cy),
                           (x + cx + cw, y + cy + ch), (255, 0, 0), 1)
            cv2.putText(display_img, ch_label, (x + cx, y + cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        self.result_var.set(plate_text if plate_text else "—")
        self.status_var.set(f"Nhận dạng xong: {len(chars)} ký tự.")
        self._show_image(display_img)

    # --------------------------------------------------------- DISPLAY ----
    def _show_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Lấy kích thước thực tế của khung hiển thị để ảnh luôn vừa khít
        self.canvas.update_idletasks()
        max_w = max(self.canvas.winfo_width() - 10, 400)
        max_h = max(self.canvas.winfo_height() - 10, 300)
        pil_img.thumbnail((max_w, max_h))
        tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.configure(image=tk_img, text="")
        self.canvas.image = tk_img  # giữ tham chiếu tránh bị garbage collected


def main():
    root = tk.Tk()
    root.option_add("*Font", "Arial 10")
    # Ép Tcl dùng UTF-8 để tránh lỗi hiển thị tiếng Việt (\u1eE8NG...) trên
    # các hệ thống WSL/Linux có locale hệ thống chưa được đặt về UTF-8.
    # try:
    #     root.tk.call("encoding", "system", "utf-8")
    # except tk.TclError:
    #     pass
    app = PlateRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()