import cv2
import os
import math
import numpy as np

from pipeline import read_chars
from utils import crop_obb

def process_video_file(input_path, plate_detector, char_detector, UPLOAD_FOLDER):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, 'Không thể đọc được video này'
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30 
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = total_frames / fps if fps > 0 else 0
    
    # Giới hạn an toàn thời lượng video để tránh nghẽn server
    if video_duration_sec > 120:
        cap.release()
        return None, f'Video quá dài ({int(video_duration_sec)}s). Chỉ hỗ trợ video dưới 2 phút.'

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
    return finalized_results, None