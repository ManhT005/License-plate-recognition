import os
import cv2
import numpy as np
import torch
import math
from difflib import SequenceMatcher # Dùng để xử lý mờ nhòe
from flask import Flask, request, jsonify, render_template, Response
from ultralytics import YOLO
from pathlib import Path

from pipeline import run_pipeline, debug_segment, debug_obb_points, crop_obb, read_chars
from realtime import RealtimeProcessor, SimpleJPEGGenerator

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load models ────────────────────────────────────────
# Sử dụng cuda:0 đại diện cho card đồ họa đầu tiên (RTX 3050)
DEVICE      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BASE_DIR    = Path(__file__).resolve().parent
PLATE_MODEL = BASE_DIR / 'models' / 'license_detection.pt'
CHAR_MODEL  = BASE_DIR / 'models' / 'yolo_ocr.pt'

# Khởi tạo mô hình và đẩy thẳng lên VRAM của GPU
plate_detector = YOLO(str(PLATE_MODEL))
plate_detector.to(DEVICE)

char_detector  = YOLO(str(CHAR_MODEL))
char_detector.to(DEVICE)

print(f"✅ Models loaded | Device: {DEVICE}")

# ── Khởi tạo Realtime Processor (Chạy ngầm) ─────────────
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
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return jsonify({'error': 'Không thể đọc được video này'}), 400
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30 
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = total_frames / fps if fps > 0 else 0
    
    # Giới hạn an toàn thời lượng video để tránh nghẽn server
    if video_duration_sec > 120:
        cap.release()
        return jsonify({'error': f'Video quá dài ({int(video_duration_sec)}s). Chỉ hỗ trợ video dưới 2 phút.'}), 400

    # Cấu hình Tracking & Buffer chuyên sâu
    step_sec = 0.2  # Tăng tần suất quét lên 5 frame/giây để bắt được khung hình rõ nhất
    current_sec = 0.0
    TIME_TO_LIVE = 1.5  # Nếu sau 1.5s không thấy lại ID này -> Xe đã đi qua, tiến hành chốt sổ
    
    # Cấu trúc lưu trữ buffer: { track_id: { 'best_text': str, 'max_len': int, 'last_seen': float, 'first_seen': float, 'best_frame': array, 'data': dict } }
    track_buffer = {}
    finalized_results = []

    print(f"🎬 Bắt đầu phân tích video sử dụng YOLO Tracker: Dài {int(video_duration_sec)}s | FPS: {fps}")

    while current_sec < video_duration_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
            
        # 🟢 Thay vì gọi run_pipeline thông thường, ta gọi trực tiếp lệnh track ở đây để duy trì luồng lưu vết
        # Bắt buộc dùng persist=True để YOLO kích hoạt ByteTrack/BoT-SORT giữ ID qua các frame
        plate_results = plate_detector.track(source=frame, conf=0.5, persist=True, device=0, imgsz=640)[0]
        
        # Kiểm tra xem có phát hiện và track được đối tượng nào không
        if plate_results.obb is not None and len(plate_results.obb) > 0:
            
            # Trích xuất danh sách Track IDs từ mô hình OBB của Ultralytics
            if plate_results.obb.id is not None:
                track_ids = plate_results.obb.id.int().cpu().tolist()
            else:
                # Nếu frame này chưa kịp gán ID, bỏ qua hoặc xử lý như không có ID
                track_ids = [None] * len(plate_results.obb)

            for i, box in enumerate(plate_results.obb.xyxyxyxy):
                tid = track_ids[i]
                if tid is None:
                    continue  # Bỏ qua các đối tượng lỗi không có ID tracking
                    
                pts = box.cpu().numpy().reshape(4, 2)
                plate_class = int(plate_results.obb.cls[i].item())
                
                # Cắt và xử lý ảnh biển số
                plate_img = crop_obb(frame, pts)
                plate_text, _ = read_chars(plate_img, char_detector, plate_class)
                
                # Lọc nhiễu ký tự quá ngắn lúc xe ở quá xa
                if not plate_text or len(plate_text) < 5:
                    continue
                
                # Tiến hành vẽ trực tiếp nhãn kèm Track ID lên khung hình để làm tư liệu lưu trữ
                pts_int = pts.astype(int)
                cls_name = 'BSD' if plate_class == 0 else 'BSV'
                label = f"ID:{tid} [{cls_name}] {plate_text}"
                
                annotated_frame = frame.copy()
                cv2.polylines(annotated_frame, [pts_int], True, (0, 255, 0), 2)
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                text_org = (pts_int[:, 0].min(), max(pts_int[:, 1].min() - 15, 20))
                cv2.rectangle(annotated_frame, (text_org[0] - 4, text_org[1] - text_size[1] - 4),
                              (text_org[0] + text_size[0] + 4, text_org[1] + 4), (0, 255, 0), cv2.FILLED)
                cv2.putText(annotated_frame, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                # Cập nhật thông tin vào trạm chờ (Buffer) dựa theo Track ID duy nhất
                if tid in track_buffer:
                    track_buffer[tid]['last_seen'] = current_sec
                    # Chiến lược: Ưu tiên chuỗi có độ dài lớn nhất (đầy đủ ký tự nhất)
                    if len(plate_text) > track_buffer[tid]['max_len']:
                        track_buffer[tid]['best_text'] = plate_text
                        track_buffer[tid]['max_len'] = len(plate_text)
                        track_buffer[tid]['best_frame'] = annotated_frame
                        track_buffer[tid]['data'] = {'index': tid, 'class': cls_name, 'text': plate_text}
                else:
                    # Khởi tạo bản ghi mới cho xe mới xuất hiện lần đầu
                    track_buffer[tid] = {
                        'best_text': plate_text,
                        'max_len': len(plate_text),
                        'last_seen': current_sec,
                        'first_seen': current_sec,
                        'best_frame': annotated_frame,
                        'data': {'index': tid, 'class': cls_name, 'text': plate_text}
                    }

        # ── HÀM CHỐT SỔ TỰ ĐỘNG (Garbage Collection & Finalize) ──
        # Duyệt qua các ID trong buffer xem có ID nào đã khuất bóng quá TIME_TO_LIVE giây không
        for tid in list(track_buffer.keys()):
            if current_sec - track_buffer[tid]['last_seen'] > TIME_TO_LIVE:
                best_data = track_buffer[tid]
                # Điều kiện chốt sổ: Chuỗi ký tự tích lũy phải đạt độ dài hợp lệ (>= 6 ký tự)
                if best_data['max_len'] >= 6:
                    mins, secs = divmod(int(best_data['first_seen']), 60)
                    timestamp = f"{mins:02d}:{secs:02d}"
                    
                    output_name = f"track_id_{tid}_frame.jpg"
                    output_path = os.path.join(UPLOAD_FOLDER, output_name)
                    cv2.imwrite(output_path, best_data['best_frame'])
                    
                    finalized_results.append({
                        'timestamp': timestamp,
                        'img_url': f"/static/{output_name}?v={np.random.randint(0,9999)}",
                        'plates': [{
                            'index': best_data['data']['index'],
                            'class': best_data['data']['class'],
                            'text': best_data['best_text']  # Chuỗi ký tự đầy đủ nhất thu thập được
                        }]
                    })
                # Giải phóng bộ nhớ buffer cho ID này
                del track_buffer[tid]
                
        current_sec += step_sec

    # Khi video kết thúc, ép chốt sổ toàn bộ những ID còn sót lại trong hàng đợi
    for tid, best_data in track_buffer.items():
        if best_data['max_len'] >= 6:
            mins, secs = divmod(int(best_data['first_seen']), 60)
            output_name = f"track_id_{tid}_frame.jpg"
            output_path = os.path.join(UPLOAD_FOLDER, output_name)
            cv2.imwrite(output_path, best_data['best_frame'])
            
            finalized_results.append({
                'timestamp': f"{mins:02d}:{secs:02d}",
                'img_url': f"/static/{output_name}?v={np.random.randint(0,9999)}",
                'plates': [{
                    'index': best_data['data']['index'],
                    'class': best_data['data']['class'],
                    'text': best_data['best_text']
                }]
            })
            
    cap.release()
    print(f"✅ Phân tích hoàn tất. Tìm thấy toàn bộ {len(finalized_results)} xe độc nhất.")
    
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)