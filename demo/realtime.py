import cv2
import numpy as np
import threading
import queue
import sys
import os
import time
from collections import deque
from pathlib import Path
from difflib import SequenceMatcher 

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import crop_obb, read_chars 

class RealtimeProcessor:
    def __init__(self, plate_detector, char_detector, source=0, fps_limit=15):
        self.plate_detector = plate_detector
        self.char_detector = char_detector
        self.source = source
        self.fps_limit = fps_limit
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=1)
        
        self.is_running = False
        self.capture_thread = None
        self.process_thread = None
        
        self.latest_result = None
        self.latest_frame = None
        self.fps = 0
        self.processing_time = 0
        
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        
        self.track_buffer = {}
        self.finalized_plates = deque(maxlen=20) 
        self.time_to_live = 2.0 
        
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.track_buffer.clear()
        self.finalized_plates.clear()
        
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
        
        self.capture_thread.start()
        self.process_thread.start()
        
    def stop(self):
        print("Đang dừng Realtime...")
        self.is_running = False 
        self.camera_released = False 
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3)
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=3)
            
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except: pass
        while not self.result_queue.empty():
            try: self.result_queue.get_nowait()
            except: pass
            
        self.track_buffer.clear()

    def _capture_frames(self):
        if sys.platform.startswith('win') and isinstance(self.source, int):
             cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
             cap = cv2.VideoCapture(self.source)
             
        if not cap.isOpened():
            print("❌ Không thể mở camera. Hãy kiểm tra xem camera có bị ứng dụng khác chiếm dụng không.")
            self.is_running = False
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"✅ Camera mở | FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(self.source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                try:
                    self.frame_queue.put(frame, timeout=0.1) 
                except queue.Full:
                    pass
        finally:
            cap.release()
            self.camera_released = True
            
    def _process_frames(self):
        try:
            while self.is_running:
                try:
                    frame = self.frame_queue.get(timeout=1)
                except queue.Empty:
                    continue
                    
                start_time = time.time()
                current_time = time.time()
                
                annotated_frame = frame.copy()
                current_frame_plates = []
                seen_tids_in_frame = set() # Bộ lọc tránh hiện 2 biển số cho cùng 1 xe trong 1 frame
                
                plate_results = self.plate_detector.track(source=frame, conf=0.5, persist=True, device=0, imgsz=640, verbose=False)[0]
                
                if plate_results.obb is not None and len(plate_results.obb) > 0:
                    if plate_results.obb.id is not None:
                        track_ids = plate_results.obb.id.int().cpu().tolist()
                    else:
                        track_ids = [None] * len(plate_results.obb)

                    for i, box in enumerate(plate_results.obb.xyxyxyxy):
                        tid = track_ids[i]
                        if tid is None:
                            continue
                            
                        pts = box.cpu().numpy().reshape(4, 2)
                        
                        # Tính tọa độ Tâm của biển số để dò vị trí
                        cx = int(np.mean(pts[:, 0]))
                        cy = int(np.mean(pts[:, 1]))
                        
                        plate_class = int(plate_results.obb.cls[i].item())
                        plate_img = crop_obb(frame, pts)
                        plate_text, _ = read_chars(plate_img, self.char_detector, plate_class)
                        
                        if not plate_text or len(plate_text) < 5:
                            continue
                            
                        cls_name = 'BSD' if plate_class == 0 else 'BSV'
                        clean_text = plate_text.replace("-", "").replace(" ", "").upper()
                        matched_tid = tid

                        if tid not in self.track_buffer:
                            # 1. Khớp bằng chữ (OCR Text)
                            for active_tid, info in self.track_buffer.items():
                                active_clean = info['best_text'].replace("-", "").replace(" ", "").upper()
                                if SequenceMatcher(None, clean_text, active_clean).ratio() >= 0.75:
                                    matched_tid = active_tid
                                    break
                                    
                            # 2. Khớp bằng Tọa Độ (Chống lóa/nhòe làm sai chữ)
                            if matched_tid == tid:
                                for active_tid, info in self.track_buffer.items():
                                    if 'center' in info:
                                        last_cx, last_cy = info['center']
                                        dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                                        # Nếu box mới xuất hiện cách box cũ dưới 90 pixel -> Nó chính là cái xe đó
                                        if dist < 90:
                                            matched_tid = active_tid
                                            break
                            
                            # 3. Khớp với xe đi qua đi lại (Vừa chốt sổ)
                            if matched_tid == tid:
                                for finalized in list(self.finalized_plates):
                                    finalized_clean = finalized['text'].replace("-", "").replace(" ", "").upper()
                                    if SequenceMatcher(None, clean_text, finalized_clean).ratio() >= 0.80:
                                        matched_tid = finalized['index']
                                        self.track_buffer[matched_tid] = {
                                            'text_history': {finalized['text']: 5}, 
                                            'best_text': finalized['text'],
                                            'max_len': len(finalized['text']),
                                            'last_seen': current_time,
                                            'first_seen': current_time - 2.0,
                                            'center': (cx, cy),
                                            'data': finalized
                                        }
                                        if finalized in self.finalized_plates:
                                            self.finalized_plates.remove(finalized)
                                        break

                        # Bộ lọc Không gian: Nếu trong cùng 1 frame, AI nhìn nhầm 1 biển thành 2 biển -> Chặn lại chỉ cho vẽ 1 cái
                        if matched_tid in seen_tids_in_frame:
                            continue
                        seen_tids_in_frame.add(matched_tid)

                        # Bầu chọn và cập nhật
                        if matched_tid in self.track_buffer:
                            self.track_buffer[matched_tid]['last_seen'] = current_time
                            self.track_buffer[matched_tid]['center'] = (cx, cy) # Liên tục cập nhật tọa độ tâm mới nhất
                            
                            history = self.track_buffer[matched_tid]['text_history']
                            history[plate_text] = history.get(plate_text, 0) + 1
                            
                            best_text = max(history.keys(), key=lambda k: (history[k], len(k)))
                            
                            self.track_buffer[matched_tid]['best_text'] = best_text
                            self.track_buffer[matched_tid]['max_len'] = len(best_text)
                            self.track_buffer[matched_tid]['data'] = {'index': matched_tid, 'class': cls_name, 'text': best_text}
                        else:
                            self.track_buffer[tid] = {
                                'text_history': {plate_text: 1},
                                'best_text': plate_text,
                                'max_len': len(plate_text),
                                'last_seen': current_time,
                                'first_seen': current_time,
                                'center': (cx, cy),
                                'data': {'index': tid, 'class': cls_name, 'text': plate_text}
                            }

                        current_frame_plates.append(self.track_buffer[matched_tid]['data'])

                        pts_int = pts.astype(int)
                        label = f"ID:{matched_tid} [{cls_name}] {self.track_buffer[matched_tid]['best_text']}"
                        
                        cv2.polylines(annotated_frame, [pts_int], True, (0, 255, 0), 2)
                        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        text_org = (pts_int[:, 0].min(), max(pts_int[:, 1].min() - 15, 20))
                        cv2.rectangle(annotated_frame, (text_org[0] - 4, text_org[1] - text_size[1] - 4),
                                      (text_org[0] + text_size[0] + 4, text_org[1] + 4), (0, 255, 0), cv2.FILLED)
                        cv2.putText(annotated_frame, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                # Dọn dẹp
                for tid in list(self.track_buffer.keys()):
                    if current_time - self.track_buffer[tid]['last_seen'] > self.time_to_live:
                        best_data = self.track_buffer[tid]
                        if best_data['max_len'] >= 6:
                            self.finalized_plates.append(best_data['data'])
                        del self.track_buffer[tid]
                
                processing_time = time.time() - start_time
                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    fps = 10 / sum(self.fps_history) if self.fps_history else 0
                    self.fps = fps
                    
                self.fps_history.append(processing_time)
                self.processing_time = processing_time
                
                self.latest_frame = annotated_frame
                self.latest_result = {
                    'plates': current_frame_plates,                  
                    'finalized_plates': list(self.finalized_plates), 
                    'frame': annotated_frame,
                    'fps': self.fps,
                    'processing_time': f"{processing_time*1000:.1f}ms",
                    'frame_count': self.frame_count
                }
                
                try: self.result_queue.put_nowait(self.latest_result)
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(self.latest_result)
                    except: pass
                        
        except Exception as e:
            print(f"❌ Error in processing: {e}")
            
    def get_latest_result(self):
        return self.latest_result
        
    def get_frame_for_stream(self, scale=1.0):
        if self.latest_frame is None:
            return None
        frame = self.latest_frame.copy()
        if scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            
        fps_text = f"FPS: {self.fps:.1f} | Processing: {self.processing_time*1000:.1f}ms | Buffer: {len(self.track_buffer)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

class VideoStreamGenerator:
    def __init__(self, processor, scale=1.0, quality=70):
        self.processor = processor
        self.scale = scale
        self.quality = quality
        
    def generate(self):
        boundary = b'--frame\r\n'
        while self.processor.is_running:
            frame = self.processor.get_frame_for_stream(self.scale)
            if frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ret: continue
            frame_bytes = buffer.tobytes()
            yield (boundary + b'Content-Type: image/jpeg\r\n' + b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n' + b'Content-Disposition: inline\r\n' + b'\r\n' + frame_bytes + b'\r\n')

class SimpleJPEGGenerator:
    def __init__(self, processor, scale=0.75, quality=80):
        self.processor = processor
        self.scale = scale
        self.quality = quality
        
    def get_jpeg(self):
        frame = self.processor.get_frame_for_stream(self.scale)
        if frame is None:
            return None
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ret: return None
        return buffer.tobytes()