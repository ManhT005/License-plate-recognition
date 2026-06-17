import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, Response
from ultralytics import YOLO
from pathlib import Path

from pipeline import run_pipeline, debug_segment, debug_obb_points
from realtime import RealtimeProcessor, SimpleJPEGGenerator

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

# ── Khởi tạo Realtime Processor (Chạy ngầm) ─────────────
# Khai báo sẵn nhưng chưa bật camera (chỉ bật khi gọi API start)
realtime_processor = RealtimeProcessor(plate_detector, char_detector, source=0)
jpeg_generator = SimpleJPEGGenerator(realtime_processor)


# ── Routes Cơ Bản ──────────────────────────────────────
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


# ── Routes Cho Realtime ────────────────────────────────
@app.route('/realtime/start', methods=['POST'])
def start_realtime():
    data = request.json or {}
    source = data.get('source', 0)
    
    realtime_processor.source = source
    if not realtime_processor.is_running:
        realtime_processor.start()
        
    return jsonify({"status": "started"}), 200


@app.route('/realtime/stop', methods=['POST'])
def stop_realtime():
    if realtime_processor.is_running:
        realtime_processor.stop()
    return jsonify({"status": "stopped"}), 200


@app.route('/video_frame')
def video_frame():
    if not realtime_processor.is_running:
        return "Camera not running", 500
        
    frame_bytes = jpeg_generator.get_jpeg()
    if frame_bytes is None:
        return "Waiting for frame", 500
        
    return Response(frame_bytes, mimetype='image/jpeg')


@app.route('/realtime/results')
def realtime_results():
    if not realtime_processor.is_running:
        return jsonify({'error': 'Not running'}), 500
        
    res = realtime_processor.get_latest_result()
    if not res:
        return jsonify({'fps': 0, 'processing_time': '0ms', 'frame_count': 0, 'plate_texts': []})
        
    # Lấy text của các biển số đưa lên Frontend
    plate_texts = [f"[{p['class']}] {p['text']}" for p in res.get('plates', [])]
    
    return jsonify({
        'fps': res.get('fps', 0),
        'processing_time': res.get('processing_time', '0ms'),
        'frame_count': res.get('frame_count', 0),
        'plate_texts': plate_texts
    })


if __name__ == '__main__':
    # use_reloader=False giúp tránh lỗi VSCode SystemExit
    # threaded=True giúp Flask xử lý song song nhiều request (VD: vừa kéo ảnh vừa kéo kết quả)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)