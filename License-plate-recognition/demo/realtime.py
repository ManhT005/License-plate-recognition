import cv2
import numpy as np
import threading
import queue
import sys
import os
from collections import deque
from pathlib import Path

# Ensure pipeline can be imported from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import run_pipeline

class RealtimeProcessor:
    def __init__(self, plate_detector, char_detector, source=0, fps_limit=15):
        """
        source: 0 for webcam, or path to video file
        fps_limit: target FPS for processing
        """
        self.plate_detector = plate_detector
        self.char_detector = char_detector
        self.source = source
        self.fps_limit = fps_limit
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=1)
        
        self.is_running = False
        self.capture_thread = None
        self.process_thread = None
        
        # Cache latest result
        self.latest_result = None
        self.latest_frame = None
        self.fps = 0
        self.processing_time = 0
        
        # Performance tracking
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        
    def start(self):
        """Start realtime processing"""
        if self.is_running:
            return
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
        
        self.capture_thread.start()
        self.process_thread.start()
        
    # Trong realtime.py

    def stop(self):
        """Stop realtime processing"""
        print("Đang dừng Realtime...")
        self.is_running = False # Đặt cờ dừng trước
        
        # Thêm biến cờ để kiểm tra xem camera đã đóng chưa
        self.camera_released = False 
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3)
            
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=3)
            
        # Xóa hàng đợi để giải phóng bộ nhớ
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except: pass
            
        while not self.result_queue.empty():
            try: self.result_queue.get_nowait()
            except: pass

    def _capture_frames(self):
        """Capture frames from source"""
        # Thêm cv2.CAP_DSHOW cho Windows nếu source=0 để mở camera nhanh hơn và tránh kẹt
        if sys.platform.startswith('win') and isinstance(self.source, int):
             cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
             cap = cv2.VideoCapture(self.source)
             
        if not cap.isOpened():
            print("❌ Không thể mở camera/video. Hãy kiểm tra xem camera có bị ứng dụng khác chiếm dụng không.")
            self.is_running = False
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"✅ Camera/Video mở | FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        try:
            while self.is_running:
                # Đọc frame. Nếu camera bị ngắt đột ngột, ret sẽ là False
                ret, frame = cap.read()
                if not ret:
                    if isinstance(self.source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("❌ Mất kết nối với Camera.")
                        break
                    
                try:
                    # Giữ nguyên logic put vào queue
                    self.frame_queue.put(frame, timeout=0.1) # Dùng timeout thay vì nowait để tránh lỗi
                except queue.Full:
                    pass
        finally:
            print("Đang giải phóng Camera...")
            cap.release()
            self.camera_released = True
            print("✅ Đã đóng Camera thành công.")
            
    def _process_frames(self):
        """Process frames from queue"""
        import time
        
        try:
            while self.is_running:
                try:
                    frame = self.frame_queue.get(timeout=1)
                except queue.Empty:
                    continue
                    
                # Process frame
                start_time = time.time()
                annotated_frame, plates_data = run_pipeline(
                    frame, self.plate_detector, self.char_detector
                )
                processing_time = time.time() - start_time
                
                # Add FPS info to frame
                self.frame_count += 1
                
                if self.frame_count % 10 == 0:
                    fps = 10 / sum(self.fps_history) if self.fps_history else 0
                    self.fps = fps
                    
                self.fps_history.append(processing_time)
                self.processing_time = processing_time
                
                # Store result
                self.latest_frame = annotated_frame
                self.latest_result = {
                    'plates': plates_data,
                    'frame': annotated_frame,
                    'fps': self.fps,
                    'processing_time': f"{processing_time*1000:.1f}ms",
                    'frame_count': self.frame_count
                }
                
                # Put result in queue (overwrite if full)
                try:
                    self.result_queue.put_nowait(self.latest_result)
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(self.latest_result)
                    except:
                        pass
                        
        except Exception as e:
            print(f"❌ Error in processing: {e}")
            
    def get_latest_result(self):
        """Get latest processing result"""
        # Không dùng result_queue.get_nowait() nữa vì nó sẽ làm rỗng queue
        # Thay vào đó chỉ trả về biến lưu trữ cache mới nhất
        return self.latest_result
        
    def get_frame_for_stream(self, scale=1.0):
        """Get frame for streaming (MJPEG)"""
        if self.latest_frame is None:
            return None
            
        frame = self.latest_frame.copy()
        
        if scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            
        # Add FPS text
        fps_text = f"FPS: {self.fps:.1f} | Processing: {self.processing_time*1000:.1f}ms"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame


class VideoStreamGenerator:
    """Generate MJPEG stream from RealtimeProcessor"""
    
    def __init__(self, processor, scale=1.0, quality=70):
        self.processor = processor
        self.scale = scale
        self.quality = quality
        
    def generate(self):
        """Yield MJPEG frames"""
        boundary = b'--frame\r\n'
        while self.processor.is_running:
            frame = self.processor.get_frame_for_stream(self.scale)
            if frame is None:
                continue
                
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ret:
                continue
            
            # Yield in MJPEG format
            frame_bytes = buffer.tobytes()
            yield (boundary +
                   b'Content-Type: image/jpeg\r\n' +
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n' +
                   b'Content-Disposition: inline\r\n' +
                   b'\r\n' +
                   frame_bytes + b'\r\n')


class SimpleJPEGGenerator:
    """Generate single JPEG frames for polling"""
    
    def __init__(self, processor, scale=0.75, quality=80):
        self.processor = processor
        self.scale = scale
        self.quality = quality
        
    def get_jpeg(self):
        """Get latest frame as JPEG bytes"""
        frame = self.processor.get_frame_for_stream(self.scale)
        if frame is None:
            return None
            
        ret, buffer = cv2.imencode('.jpg', frame, 
                                  [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ret:
            return None
            
        return buffer.tobytes()
