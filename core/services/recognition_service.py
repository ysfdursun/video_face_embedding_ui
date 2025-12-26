# -*- coding: utf-8 -*-
"""
Recognition Service - Video Face Recognition

Based on user's working demo code.
Uses avg_embeddings.pkl database for face recognition.
"""
import cv2
import numpy as np
import os
import pickle
import time
from collections import defaultdict
from django.conf import settings

# Global models (lazy loaded)
_app = None
_rec = None
_db = None

THRESHOLD_DEFAULT = 0.27

def norm_crop(img, landmark, image_size=112):
    """Align and crop face using landmarks."""
    from insightface.utils import face_align
    M = face_align.estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def get_models():
    """Initialize InsightFace models."""
    global _app, _rec
    
    if _app is not None and _rec is not None:
        return _app, _rec
    
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    
    print("🚀 Loading Recognition Models...")
    
    # Detection model
    _app = FaceAnalysis(name='buffalo_sc', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    _app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Recognition model
    root = os.path.join(os.path.expanduser("~"), ".insightface")
    path = os.path.join(root, "models", "buffalo_l", "w600k_r50.onnx")
    _rec = get_model(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    _rec.prepare(ctx_id=0)
    
    print("✅ Models loaded successfully")
    return _app, _rec

def get_db():
    """Load embeddings database."""
    global _db
    
    if _db is not None:
        return _db
    
    db_path = os.path.join(settings.BASE_DIR, "avg_embeddings.pkl")
    print(f"📂 Loading DB: {os.path.basename(db_path)}")
    
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        return {}
    
    with open(db_path, 'rb') as f:
        data = pickle.load(f)
    
    db = {}
    for p, info in data.items():
        if isinstance(info, list):
            embs = np.array(info)
        elif isinstance(info, dict) and 'embeddings' in info:
            embs = np.array(info['embeddings'])
        else:
            embs = np.array([info])
        
        if embs.ndim == 3:
            embs = embs.reshape(embs.shape[0], -1)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        if len(embs) == 0:
            continue
        
        mean = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            db[p] = mean / norm
    
    print(f"   Loaded {len(db)} identities")
    _db = db
    return db

def process_recognition_stream(video_path, threshold=0.27, min_face_size=30, frame_skip=5):
    """
    Process video for face recognition and yield MJPEG frames.
    Based on user's working demo code.
    """
    print(f"\n{'='*50}")
    print(f"🎬 Recognition Starting")
    print(f"   Video: {video_path}")
    print(f"   Threshold: {threshold}, MinFace: {min_face_size}, FrameSkip: {frame_skip}")
    print(f"{'='*50}\n")
    
    # Yield initial loading frame immediately to start stream
    loading_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(loading_frame, "Modeller Yukleniyor...", (150, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    _, buf = cv2.imencode('.jpg', loading_frame)
    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    
    # Now load models
    app, rec = get_models()
    db = get_db()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        # Error frame
        err = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err, "VIDEO ACILAMADI", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buf = cv2.imencode('.jpg', err)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {video_fps}fps, {total_frames} frames")
    
    frame_count = 0
    start_time = time.time()
    fps_update_time = start_time
    current_fps = 0
    
    person_frames = defaultdict(int)
    last_results = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # FPS calculation
        if frame_count % 10 == 0:
            now = time.time()
            elapsed = now - fps_update_time
            if elapsed > 0:
                current_fps = 10 / elapsed
            fps_update_time = now
        
        # Recognition every N frames - ONLY yield on these frames
        if frame_count % frame_skip == 0:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            faces = app.get(small)
            last_results = []
            
            for face in faces:
                bbox = face.bbox.astype(int)
                if (bbox[3] - bbox[1]) < (min_face_size / 2):
                    continue
                
                # Align
                aimg = norm_crop(small, face.kps)
                feat = rec.get_feat(aimg).flatten()
                feat = feat / np.linalg.norm(feat)
                
                # Match
                max_score = -1
                name = "Unknown"
                
                for p, temp in db.items():
                    score = float(np.dot(temp, feat))
                    if score > max_score:
                        max_score = score
                        name = p
                
                real_bbox = bbox * 2
                
                if max_score > threshold:
                    color = (0, 255, 0)
                    label = f"{name} ({max_score:.2f})"
                    person_frames[name] += 1
                else:
                    color = (0, 0, 255)
                    label = f"Unknown ({max_score:.2f})"
                    person_frames["Unknown"] += 1
                
                last_results.append((real_bbox, label, color))
            
            # Draw results
            for real_bbox, label, color in last_results:
                cv2.rectangle(frame, (real_bbox[0], real_bbox[1]), (real_bbox[2], real_bbox[3]), color, 2)
                cv2.putText(frame, label, (real_bbox[0], real_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Info overlay
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            info_text = f"FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            active_text = f"Aktif: {len(last_results)} kisi"
            cv2.putText(frame, active_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Resize for streaming (reduce bandwidth)
            if width > 960:
                scale = 960 / width
                stream_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            else:
                stream_frame = frame
            
            # Encode with lower quality for faster streaming
            _, buffer = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()
    
    # Statistics
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print("📊 RECOGNITION COMPLETE")
    print(f"Total Frames: {frame_count}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Average FPS: {frame_count/total_time:.1f}" if total_time > 0 else "")
    
    sorted_persons = sorted(person_frames.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_persons[:15]:
        duration = (count * frame_skip) / video_fps
        print(f"   {name}: {count} times (~{duration:.1f}s)")
    print(f"{'='*50}")

def get_identities_count():
    """Return number of identities in database."""
    db = get_db()
    return len(db) if db else 0
