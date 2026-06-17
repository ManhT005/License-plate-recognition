import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


# ── Bước 1: Tiền xử lý biển số ────────────────────────
def preprocess_plate(plate_img):
    # ── 1. Resize ─────────────────────────────────────
    TARGET_H = 120
    scale    = TARGET_H / plate_img.shape[0]
    target_w = int(plate_img.shape[1] * scale)
    plate    = cv2.resize(plate_img, (target_w, TARGET_H),
                          interpolation=cv2.INTER_CUBIC)

    # ── 2. Grayscale + CLAHE ──────────────────────────
    gray  = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    # ── 3. Sharpen ────────────────────────────────────
    sharpen_k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    sharp     = cv2.filter2D(gray, -1, sharpen_k)

    # ── 4. Sobel edge ─────────────────────────────────
    sobelx = cv2.Sobel(sharp, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(sharp, cv2.CV_64F, 0, 1, ksize=3)
    sobel  = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))

    combined = cv2.addWeighted(sharp, 0.7, sobel, 0.3, 0)

    # ── 5. Threshold kép ──────────────────────────────
    # Otsu global
    blur      = cv2.GaussianBlur(combined, (3, 3), 0)
    _, otsu   = cv2.threshold(blur, 0, 255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive để bắt ký tự ở vùng sáng/tối không đều
    adaptive  = cv2.adaptiveThreshold(blur, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 21, 8)

    # Kết hợp: lấy giao (AND) → chỉ giữ vùng cả 2 đều đồng ý là ký tự
    binary    = cv2.bitwise_and(otsu, adaptive)

    # ── 6. Morphology CẨN THẬN ────────────────────────
    # Close nhỏ: lấp lỗ hổng bên trong ký tự
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=1)

    # Erosion: mở rộng khoảng đen giữa các ký tự
    # → tách ký tự gần nhau như '88'
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary  = cv2.erode(binary, erode_k, iterations=1)

    # Open: loại noise nhỏ
    open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k, iterations=1)

    return plate, binary


# ── Bước 2: Tách 2 dòng cho biển BSV ──────────────────
def detect_rows(binary):
    h_proj    = np.sum(binary, axis=1)
    threshold = h_proj.max() * 0.15

    rows, in_row, start = [], False, 0
    for i, val in enumerate(h_proj):
        if val > threshold and not in_row:
            in_row, start = True, i
        elif val <= threshold and in_row:
            in_row = False
            if i - start > 10:
                rows.append((start, i))
    return rows


# ── Bước 3: Tách ký tự trong 1 dòng ───────────────────
def segment_row(binary_row, plate_row):
    h = binary_row.shape[0]

    contours, _ = cv2.findContours(
        binary_row, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for cnt in contours:
        x, y, w, h_c = cv2.boundingRect(cnt)

        # Filter theo chiều cao
        if h_c < h * 0.30:           continue   # giảm từ 0.35 để không bỏ sót
        if w < 4:                     continue
        if w / float(h_c) > 1.3:     continue   # giảm từ 1.5 để lọc '88' merged

        area = cv2.contourArea(cnt)
        if area < h * 6:              continue   # lọc dấu . và -

        # Nếu box quá rộng so với ký tự đơn → thử tách đôi
        if w / float(h_c) > 0.9 and w > h_c * 1.1:
            mid = x + w // 2
            boxes.append((x,   y, mid - x, h_c))
            boxes.append((mid, y, x + w - mid, h_c))
        else:
            boxes.append((x, y, w, h_c))

    # Fallback vertical projection
    if len(boxes) < 2:
        v_proj    = np.sum(binary_row, axis=0)
        threshold = v_proj.max() * 0.15
        in_c, segs, start = False, [], 0
        for i, val in enumerate(v_proj):
            if val > threshold and not in_c:
                in_c, start = True, i
            elif val <= threshold and in_c:
                in_c = False
                if i - start > 6:
                    segs.append((start, 0, i - start, h))
        boxes = segs

    if not boxes:
        return []

    # Merge chỉ khi gap thực sự nhỏ (< 5px)
    boxes  = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]
    for box in boxes[1:]:
        prev = merged[-1]
        gap  = box[0] - (prev[0] + prev[2])
        if gap < 5:   # giảm từ 10 xuống 5 để tránh merge ký tự hợp lệ
            nx = min(prev[0], box[0])
            ny = min(prev[1], box[1])
            nw = max(prev[0]+prev[2], box[0]+box[2]) - nx
            nh = max(prev[1]+prev[3], box[1]+box[3]) - ny
            merged[-1] = [nx, ny, nw, nh]
        else:
            merged.append(list(box))

    # Crop từ binary
    chars = []
    for x, y, w, h_c in merged:
        char_bin = binary_row[y:y+h_c, x:x+w]
        char_bgr = cv2.cvtColor(char_bin, cv2.COLOR_GRAY2BGR)
        chars.append(char_bgr)

    return chars


# ── Bước 4: Chuẩn hóa ký tự ───────────────────────────
def normalize_char(char_img, size=64):
    """Nhận ảnh BGR (từ binary) → chuẩn hóa về 64x64"""
    h, w     = char_img.shape[:2]
    scale    = (size - 8) / max(h, w)
    new_w    = max(1, int(w * scale))
    new_h    = max(1, int(h * scale))
    resized  = cv2.resize(char_img, (new_w, new_h),
                          interpolation=cv2.INTER_CUBIC)

    canvas        = np.zeros((size, size, 3), dtype=np.uint8)
    y_off         = (size - new_h) // 2
    x_off         = (size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

# ── Crop OBB ───────────────────────────────────────────
def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
    pts  = pts[np.argsort(pts[:, 1])]   # sort theo y
    top  = pts[:2][np.argsort(pts[:2, 0])]   # 2 điểm trên: sort theo x
    bot  = pts[2:][np.argsort(pts[2:, 0])]   # 2 điểm dưới: sort theo x
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)
    # thứ tự: TL, TR, BR, BL

def crop_obb(image, pts):
    pts = order_points(np.array(pts, dtype=np.float32))
    
    # Tính width, height thực của biển
    w = int(max(
        np.linalg.norm(pts[0] - pts[1]),  # top edge
        np.linalg.norm(pts[3] - pts[2])   # bottom edge
    ))
    h = int(max(
        np.linalg.norm(pts[0] - pts[3]),  # left edge
        np.linalg.norm(pts[1] - pts[2])   # right edge
    ))

    # Đảm bảo tỉ lệ hợp lý cho biển số
    # Biển VN thường w/h ~ 3:1 (1 dòng) hoặc 2:1 (2 dòng)
    if w < h:
        w, h = h, w  # swap nếu bị ngược

    w = max(w, 100)
    h = max(h, 40)

    dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (w, h))

# ── Segment theo class BSD/BSV ─────────────────────────
def preprocess_plate(plate_img):
    # ── 1. Resize chuẩn ───────────────────────────────
    TARGET_H = 120
    scale    = TARGET_H / plate_img.shape[0]
    target_w = int(plate_img.shape[1] * scale)
    plate    = cv2.resize(plate_img, (target_w, TARGET_H),
                          interpolation=cv2.INTER_CUBIC)

    # ── 2. Grayscale + CLAHE ──────────────────────────
    gray  = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    # ── 3. Tích chập Sharpening kernel ───────────────
    # Làm nét cạnh ký tự trước khi threshold
    sharpen_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    sharp = cv2.filter2D(gray, -1, sharpen_kernel)

    # ── 4. Sobel edge để tăng cạnh ký tự ─────────────
    sobelx = cv2.Sobel(sharp, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(sharp, cv2.CV_64F, 0, 1, ksize=3)
    sobel  = cv2.magnitude(sobelx, sobely)
    sobel  = cv2.convertScaleAbs(sobel)

    # ── 5. Kết hợp gray + sobel để giữ fill + edge ───
    combined = cv2.addWeighted(sharp, 0.7, sobel, 0.3, 0)

    # ── 6. Threshold ──────────────────────────────────
    blur      = cv2.GaussianBlur(combined, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── 7. Morphology: lấp lỗ + loại noise ───────────
    # Lấp lỗ hổng trong thân ký tự
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=2)

    # Loại noise nhỏ
    open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k, iterations=1)

    return plate, binary

def segment_chars(plate_img, plate_class):
    """
    plate_class: 0 = BSD (1 dòng) | 1 = BSV (2 dòng)
    """
    plate, binary = preprocess_plate(plate_img)

    if plate_class == 0:
        rows = [(0, binary.shape[0])]
    else:
        rows = detect_rows(binary)
        if len(rows) != 2:
            mid  = binary.shape[0] // 2
            rows = [(0, mid), (mid, binary.shape[0])]

    all_chars = []
    for y1, y2 in rows:
        binary_row = binary[y1:y2, :]
        plate_row  = plate[y1:y2, :]
        chars      = segment_row(binary_row, plate_row)
        chars      = [normalize_char(c) for c in chars if c.size > 0]
        all_chars.extend(chars)

    return all_chars


# ── Nhận diện 1 ký tự ─────────────────────────────────
def predict_char(char_img, char_model, char_tf, classes, device):
    if char_img.shape[0] < 10 or char_img.shape[1] < 5:
        return '?', 0.0

    pil_img = Image.fromarray(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
    tensor  = char_tf(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(char_model(tensor), dim=1)
        top1  = probs.argmax(1).item()
        conf  = probs[0][top1].item()

    if conf < 0.4:
        return '?', conf

    return classes[top1], conf


# ── Pipeline chính ─────────────────────────────────────
def run_pipeline(image_bgr, plate_detector, char_model, char_tf, classes, device):
    plate_results = plate_detector(image_bgr, conf=0.5)[0]

    if len(plate_results.obb) == 0:
        return image_bgr, []

    plates_data = []
    for i, box in enumerate(plate_results.obb.xyxyxyxy):
        pts         = box.cpu().numpy().reshape(4, 2)
        plate_class = int(plate_results.obb.cls[i].item())  # 0=BSD, 1=BSV
        plate_img   = crop_obb(image_bgr, pts)

        char_imgs  = segment_chars(plate_img, plate_class)
        plate_text = ''
        for char_img in char_imgs:
            char, conf = predict_char(char_img, char_model, char_tf, classes, device)
            plate_text += char
        plate_text = plate_text.strip('?')

        # Vẽ lên ảnh
        pts_int  = pts.astype(int)
        cls_name = 'BSD' if plate_class == 0 else 'BSV'
        label    = f"[{cls_name}] {plate_text}"

        cv2.polylines(image_bgr, [pts_int], True, (0, 255, 0), 2)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        text_org  = (pts_int[:,0].min(), max(pts_int[:,1].min() - 15, 20))
        cv2.rectangle(image_bgr,
                      (text_org[0]-4, text_org[1]-text_size[1]-4),
                      (text_org[0]+text_size[0]+4, text_org[1]+4),
                      (0, 255, 0), cv2.FILLED)
        cv2.putText(image_bgr, label, text_org,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        plates_data.append({
            'index' : i + 1,
            'class' : cls_name,
            'text'  : plate_text,
        })

    return image_bgr, plates_data