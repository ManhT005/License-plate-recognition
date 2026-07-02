import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

from utils import crop_obb, preprocess_plate_image, _dedupe_results
from config import CLASSES, STANDARD_CHARS, SKIPPED_CHARS



def map_char_name(char_detector, cls):
    names = None
    if hasattr(char_detector, 'names'):
        names = char_detector.names
    elif hasattr(char_detector, 'model') and hasattr(char_detector.model, 'names'):
        names = char_detector.model.names

    # Check if the model has a buggy 37-class names dict with ':' at index 10
    # If so, we override it and use STANDARD_CHARS directly, because the model
    # actually predicts A at index 10, B at index 11, etc.
    if isinstance(names, dict) and len(names) == 37 and names.get(10) == ':':
        names = {i: STANDARD_CHARS[i] for i in range(len(STANDARD_CHARS))}

    char_name = None
    if isinstance(names, dict):
        char_name = names.get(cls)
    elif isinstance(names, (list, tuple)):
        if 0 <= cls < len(names):
            char_name = names[cls]

    if char_name is None and 0 <= cls < len(CLASSES):
        char_name = CLASSES[cls]

    if char_name is None:
        return None

    char_name = str(char_name)
    if char_name in SKIPPED_CHARS:
        return None
    return char_name




# ── Đọc ký tự từ biển đã crop ─────────────────────────
def _chars_from_results(results, char_detector):
    boxes = _dedupe_results(results)
    chars     = []
    char_info = []
    total_conf = 0.0

    for box in boxes:
        x1   = box.xyxy[0][0].item()
        y1   = box.xyxy[0][1].item()
        x2   = box.xyxy[0][2].item()
        y2   = box.xyxy[0][3].item()
        cls  = int(box.cls[0].item())
        conf = box.conf[0].item()

        char_name = map_char_name(char_detector, cls)
        if char_name is None:
            continue

        chars.append((x1, y1, char_name, conf))
        char_info.append({
            'x1'  : int(x1), 'y1': int(y1),
            'x2'  : int(x2), 'y2': int(y2),
            'char': char_name,
            'conf': f'{conf:.2%}'
        })
        total_conf += conf

    if not chars:
        return '', [], 0, 0.0

    avg_conf = total_conf / len(chars)
    return chars, char_info, len(chars), avg_conf


def _build_plate_text(chars, plate_class):
    if plate_class == 1:  # BSV: 2 dòng
        avg_y = sum(c[1] for c in chars) / len(chars)
        row1  = sorted([c for c in chars if c[1] < avg_y], key=lambda c: c[0])
        row2  = sorted([c for c in chars if c[1] >= avg_y], key=lambda c: c[0])
        text1 = ''.join(c[2] for c in row1)
        text2 = ''.join(c[2] for c in row2)
        return f"{text1}-{text2}" if text1 and text2 else text1 or text2

    chars = sorted(chars, key=lambda c: c[0])
    return ''.join(c[2] for c in chars)


def _choose_best_result(raw_data, proc_data):
    raw_chars, raw_info, raw_n, raw_conf = raw_data
    proc_chars, proc_info, proc_n, proc_conf = proc_data

    if proc_n == 0:
        return raw_chars, raw_info
    if raw_n == 0:
        return proc_chars, proc_info

    raw_has_letter = any(c[2].isalpha() for c in raw_chars)
    proc_has_letter = any(c[2].isalpha() for c in proc_chars)

    if raw_has_letter and not proc_has_letter:
        return raw_chars, raw_info
    if proc_has_letter and raw_has_letter:
        # Prefer the result with fewer duplicate boxes when confidence is close.
        if proc_n < raw_n and proc_conf + 0.05 >= raw_conf:
            return proc_chars, proc_info
        if raw_n < proc_n and raw_conf + 0.05 >= proc_conf:
            return raw_chars, raw_info
    if raw_n > proc_n and raw_conf > 0.25:
        return raw_chars, raw_info
    if raw_n == proc_n and raw_conf > proc_conf + 0.05:
        return raw_chars, raw_info

    return proc_chars, proc_info


def read_chars(plate_img, char_detector, plate_class, conf=0.20):
    processed = preprocess_plate_image(plate_img)

    proc_results = char_detector(processed, conf=conf)[0]
    raw_results  = char_detector(plate_img, conf=conf)[0]

    proc_chars, proc_info, proc_n, proc_conf = _chars_from_results(proc_results, char_detector)
    raw_chars, raw_info, raw_n, raw_conf    = _chars_from_results(raw_results, char_detector)

    if proc_n == 0 and raw_n == 0:
        return '', []

    chars, char_info = _choose_best_result((raw_chars, raw_info, raw_n, raw_conf),
                                          (proc_chars, proc_info, proc_n, proc_conf))
    text = _build_plate_text(chars, plate_class)
    return text, char_info


# ── Pipeline chính ─────────────────────────────────────
def run_pipeline(image_bgr, plate_detector, char_detector):
   # Ép dùng GPU (device=0) và giới hạn ảnh gốc ở 640px để tránh sập VRAM 4GB
    plate_results = plate_detector.predict(source=image_bgr, conf=0.5, device=0, imgsz=640)[0]
    if len(plate_results.obb) == 0:
        return image_bgr, []

    plates_data = []
    for i, box in enumerate(plate_results.obb.xyxyxyxy):
        pts         = box.cpu().numpy().reshape(4, 2)
        plate_class = int(plate_results.obb.cls[i].item())
        plate_img   = crop_obb(image_bgr, pts)

        plate_text, _ = read_chars(plate_img, char_detector, plate_class)

        # Vẽ lên ảnh
        pts_int  = pts.astype(int)
        cls_name = 'BSD' if plate_class == 0 else 'BSV'
        label    = f"[{cls_name}] {plate_text}"

        cv2.polylines(image_bgr, [pts_int], True, (0,255,0), 2)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        text_org  = (pts_int[:,0].min(), max(pts_int[:,1].min()-15, 20))
        cv2.rectangle(image_bgr,
                      (text_org[0]-4, text_org[1]-text_size[1]-4),
                      (text_org[0]+text_size[0]+4, text_org[1]+4),
                      (0,255,0), cv2.FILLED)
        cv2.putText(image_bgr, label, text_org,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

        plates_data.append({
            'index' : i + 1,
            'class' : cls_name,
            'text'  : plate_text,
        })

    return image_bgr, plates_data


# ── Debug helpers ──────────────────────────────────────
def debug_segment(image_bgr, plate_detector, char_detector, upload_folder):
    plate_results = plate_detector(image_bgr, conf=0.5)[0]
    if len(plate_results.obb) == 0:
        return None, 'Không detect được biển'

    debug_info = []
    for i, box in enumerate(plate_results.obb.xyxyxyxy):
        pts         = box.cpu().numpy().reshape(4, 2)
        plate_class = int(plate_results.obb.cls[i].item())
        plate_img   = crop_obb(image_bgr, pts)
        plate_img_proc = preprocess_plate_image(plate_img)

        # Lưu plate crop
        plate_fname = f'plate_{i}.jpg'
        cv2.imwrite(f'{upload_folder}/{plate_fname}', plate_img)

        # Lưu plate crop đã xử lý
        proc_fname = f'plate_processed_{i}.jpg'
        cv2.imwrite(f'{upload_folder}/{proc_fname}', plate_img_proc)

        # Đọc ký tự
        plate_text, char_info = read_chars(plate_img, char_detector, plate_class)

        # Lưu ảnh debug plate
        plate_debug = plate_img.copy()
        for c in char_info:
            cv2.rectangle(plate_debug,
                          (c['x1'], c['y1']), (c['x2'], c['y2']),
                          (0,255,0), 1)
            cv2.putText(plate_debug, c['char'],
                        (c['x1'], max(c['y1']-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        debug_fname = f'plate_debug_{i}.jpg'
        cv2.imwrite(f'{upload_folder}/{debug_fname}', plate_debug)

        debug_chars = []
        for j, c in enumerate(char_info):
            x1, y1, x2, y2 = c['x1'], c['y1'], c['x2'], c['y2']
            char_crop = plate_img[max(y1-6, 0):min(y2+6, plate_img.shape[0]),
                                  max(x1-6, 0):min(x2+6, plate_img.shape[1])]
            char_name = f'plate_{i}_char_{j}.jpg'
            cv2.imwrite(f'{upload_folder}/{char_name}', char_crop)
            debug_chars.append({
                'path' : f'/static/{char_name}?v={np.random.randint(10000)}',
                'pred' : c['char'],
                'conf' : c['conf'],
                'size' : f"{x2-x1}x{y2-y1}"
            })

        debug_info.append({
            'plate_url'   : f'/static/{plate_fname}?v={np.random.randint(10000)}',
            'binary_url'  : f'/static/{proc_fname}?v={np.random.randint(10000)}',
            'plate_class' : 'BSD' if plate_class == 0 else 'BSV',
            'plate_text'  : plate_text,
            'num_chars'   : len(char_info),
            'chars'       : debug_chars
        })

    return debug_info, None


def debug_obb_points(image_bgr, plate_detector, upload_folder):
    plate_results = plate_detector(image_bgr, conf=0.5)[0]

    colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
    for box in plate_results.obb.xyxyxyxy:
        pts = box.cpu().numpy().reshape(4, 2).astype(int)
        for j, (pt, color) in enumerate(zip(pts, colors)):
            cv2.circle(image_bgr, tuple(pt), 8, color, -1)
            cv2.putText(image_bgr, str(j), (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.polylines(image_bgr, [pts], True, (255,255,0), 2)

    import numpy as np
    out_path = f'{upload_folder}/debug_pts.jpg'
    cv2.imwrite(out_path, image_bgr)
    return f'/static/debug_pts.jpg?v={np.random.randint(9999)}'