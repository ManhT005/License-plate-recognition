import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from ultralytics import YOLO
from config import UPLOAD_FOLDER, DEVICE, PLATE_MODEL, CHAR_MODEL
from pipeline import run_pipeline, debug_segment, debug_obb_points
from realtime import RealtimeProcessor, SimpleJPEGGenerator
from video_processor import process_video_file

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load models ────────────────────────────────────────
plate_detector = YOLO(str(PLATE_MODEL))
plate_detector.to(DEVICE)

char_detector  = YOLO(str(CHAR_MODEL))
char_detector.to(DEVICE)

print(f"✅ Models loaded | Device: {DEVICE}")

# ── Khởi tạo Realtime Processor ────────────────────────
realtime_processor = RealtimeProcessor(plate_detector, char_detector, source=0)
jpeg_generator = SimpleJPEGGenerator(realtime_processor)



# ── Cấu hình cho phép đọc CSS/JS từ thư mục templates ──
@app.route('/templates/<path:filename>')
def serve_templates_static(filename):
    return send_from_directory('templates', filename)


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

    # Thêm bộ lọc cho ảnh tĩnh để loại bỏ nhiễu rác
    valid_plates = []
    for p in plates_data:
        if p.get('text') and len(p['text']) >= 6:
            valid_plates.append(p)
    plates_data = valid_plates

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


@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Không tìm thấy video'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn video'}), 400
    
    input_path = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
    file.save(input_path)
    # Hứng cả kết quả và lỗi từ file video_processor
    finalized_results, error = process_video_file(input_path, plate_detector, char_detector, UPLOAD_FOLDER)
    # Nếu video_processor báo lỗi, trả về lỗi 400
    if error:
        return jsonify({'error': error}), 400
    if not finalized_results:
        return jsonify({'error': 'Không phát hiện hoặc trích xuất thành công biển số nào đủ độ dài tiêu chuẩn.'}) 
    return jsonify({
        'total_unique': len(finalized_results),
        'results': finalized_results
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
        return jsonify({
            'fps': 0, 
            'processing_time': '0ms', 
            'frame_count': 0, 
            'active_plates': [], 
            'finalized_plates': []
        })
        
    # 1. Danh sách xe ĐANG xuất hiện trong khung hình (Realtime quét liên tục)
    active_plates = [f"ID:{p['index']} [{p['class']}] {p['text']}" for p in res.get('plates', [])]
    
    # 2. Danh sách xe ĐÃ CHỐT SỔ (Lịch sử chuẩn xác đã lưu lại sau khi xe đi qua)
    finalized_plates = [f"ID:{p['index']} [{p['class']}] {p['text']}" for p in res.get('finalized_plates', [])]
    
    return jsonify({
        'fps': res.get('fps', 0),
        'processing_time': res.get('processing_time', '0ms'),
        'frame_count': res.get('frame_count', 0),
        'active_plates': active_plates,
        'finalized_plates': finalized_plates
    })


@app.route('/realtime/clear_history', methods=['POST'])
def clear_history():
    if realtime_processor:
        realtime_processor.clear_history()
    return jsonify({"status": "cleared"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)