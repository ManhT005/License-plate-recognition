import os
import torch
from pathlib import Path

# ── Định tuyến thư mục ─────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static')

# ── Cấu hình Model YOLO ────────────────────────────────
DEVICE        = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PLATE_MODEL   = BASE_DIR / 'models' / 'license_detection.pt'
CHAR_MODEL    = BASE_DIR / 'models' / 'yolo_ocr.pt'

# ── Khai báo Classes cho Model OCR ─────────────────────
CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
]
STANDARD_CHARS = CLASSES.copy()

# Cấu hình bỏ qua các ký tự gây nhiễu (ví dụ: dấu hai chấm trên biển số VN)
SKIPPED_CHARS = {':', 'colon'}