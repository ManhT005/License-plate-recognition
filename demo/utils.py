import cv2
import numpy as np


# ── Crop OBB ───────────────────────────────────────────
def order_points(pts):
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bot = pts[2:][np.argsort(pts[2:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)


def crop_obb(image, pts):
    pts = order_points(np.array(pts, dtype=np.float32))
    w   = int(max(np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[3]-pts[2])))
    h   = int(max(np.linalg.norm(pts[0]-pts[3]), np.linalg.norm(pts[1]-pts[2])))
    if w < h:
        w, h = h, w
    w = max(w, 100)
    h = max(h, 40)
    dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (w, h))


def preprocess_plate_image(plate_img, contrast=1.4, clip_limit=2.0, sharpen=True):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel)
    gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    if area1 + area2 - inter_area <= 0:
        return 0.0
    return inter_area / (area1 + area2 - inter_area)


def _dedupe_results(results, iou_thresh=0.3):
    boxes = []
    scores = []
    for box in results.boxes:
        x1 = float(box.xyxy[0][0].item())
        y1 = float(box.xyxy[0][1].item())
        x2 = float(box.xyxy[0][2].item())
        y2 = float(box.xyxy[0][3].item())
        boxes.append((x1, y1, x2, y2))
        scores.append(float(box.conf[0].item()))

    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    for i in order:
        if all(_iou(boxes[i], boxes[j]) < iou_thresh for j in keep):
            keep.append(i)

    return [results.boxes[i] for i in keep]