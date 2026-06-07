import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from pathlib import Path

from pipeline import run_pipeline, debug_segment, debug_obb_points

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load models ────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR    = Path(__file__).resolve().parent
PLATE_MODEL = BASE_DIR / 'models' / 'license_detection.pt'
CHAR_MODEL  = BASE_DIR / 'models' / 'yolo_ocr.pt'

plate_detector = YOLO(str(PLATE_MODEL))
char_detector  = YOLO(str(CHAR_MODEL))
print(f"✅ Models loaded | Device: {DEVICE}")


# ── Routes ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy ảnh'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn ảnh'}), 400

    input_path = os.path.join(UPLOAD_FOLDER, 'input.jpg')
    file.save(input_path)
    image_bgr = cv2.imread(input_path)

    annotated, plates_data = run_pipeline(
        image_bgr, plate_detector, char_detector
    )

    if not plates_data:
        return jsonify({
            'result_text' : '❌ Không phát hiện biển số',
            'img_url'     : '/static/input.jpg',
            'plates'      : []
        })

    output_path = os.path.join(UPLOAD_FOLDER, 'output.jpg')
    cv2.imwrite(output_path, annotated)

    return jsonify({
        'result_text' : ' | '.join([
            f"Biển {p['index']} ({p['class']}): {p['text']}"
            for p in plates_data
        ]),
        'img_url'     : f"/static/output.jpg?v={np.random.randint(0,9999)}",
        'plates'      : plates_data
    })


@app.route('/debug', methods=['POST'])
def debug():
    if 'image' not in request.files:
        return jsonify({'error': 'Chưa chọn ảnh'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn ảnh'}), 400

    input_path = os.path.join(UPLOAD_FOLDER, 'debug_input.jpg')
    file.save(input_path)
    image_bgr = cv2.imread(input_path)

    debug_info, err = debug_segment(
        image_bgr, plate_detector, char_detector, UPLOAD_FOLDER
    )
    if err:
        return jsonify({'error': err})
    return jsonify(debug_info)


@app.route('/debug_pts', methods=['POST'])
def debug_pts():
    if 'image' not in request.files:
        return jsonify({'error': 'Chưa chọn ảnh'}), 400
    file = request.files['image']
    input_path = os.path.join(UPLOAD_FOLDER, 'dbg_pts.jpg')
    file.save(input_path)
    image_bgr = cv2.imread(input_path)

    img_url = debug_obb_points(image_bgr, plate_detector, UPLOAD_FOLDER)
    return jsonify({'img_url': img_url})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)