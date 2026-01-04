# -*- coding: utf-8 -*-
"""
Video Face Recognizer (Pickle Integration)
==========================================
Real-time face recognition from video files.
Based strictly on user-provided VideoFaceRecognizer logic.
"""

import os
import cv2
import numpy as np
import time
import pickle
from collections import defaultdict, deque

from core.config import Config
from core.model_loader import load_detector, load_recognizer
from core.quality.face_quality import FaceQualityScorer

# Color palette for different persons
COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 255, 0),  # Lime
    (255, 128, 0),  # Orange
]

class VideoFaceRecognizer:
    """
    Real-time face recognition from video.
    """
    
    def __init__(self, threshold=None, temporal_buffer_size=20):
        """
        Initialize recognizer.
        """
        self.threshold = threshold or Config.RECOGNITION_THRESHOLD
        self.temporal_buffer_size = temporal_buffer_size
        
        print("\n" + "="*60)
        print("🎬 VIDEO FACE RECOGNIZER (Pickle Mode)")
        print("="*60)
        
        # Load embeddings database from 'avg_embeddings.pkl'
        # Can be absolute or relative path. User said: "veri tabanı bu : avg_embeddings.pkl"
        # We assume it's in the project root or we can look for it.
        candidates = [
            'avg_embeddings.pkl',
            os.path.join(Config.BASE_DIR, 'avg_embeddings.pkl'),
            r'c:\Users\hamza\OneDrive\Belgeler\GitHub\video_face_embedding_ui\avg_embeddings.pkl'
        ]
        
        self.database = {}
        loaded = False
        for path in candidates:
            if os.path.exists(path):
                print(f"📂 Loading embeddings from: {path}")
                try:
                    with open(path, 'rb') as f:
                        self.database = pickle.load(f)
                    print(f"   ✓ Loaded {len(self.database)} persons from pickle.")
                    loaded = True
                    break
                except Exception as e:
                    print(f"   ❌ Error loading pickle {path}: {e}")

        if not loaded:
            print("   ⚠️ No embeddings found in 'avg_embeddings.pkl'. Recognition will be limited.")
        
        # Load models
        self.detector = load_detector()
        self.recognizer = load_recognizer()
        
        if self.recognizer is None:
            raise RuntimeError("Recognition model not loaded!")
        
        print(f"\n🎯 Threshold: {self.threshold}")
        print(f"🔄 Temporal Buffer: {self.temporal_buffer_size} frames")
        
        self.person_colors = {}
        self.color_idx = 0
        
        self.stats = defaultdict(int)
        self.quality_stats = []
        
        self.quality_scorer = FaceQualityScorer()
        self.min_quality_threshold = 0.3
        
        self.face_trackers = {}
        self.next_face_id = 0
        self.face_association_threshold = 0.5 

    def get_color(self, person_name):
        if person_name not in self.person_colors:
            self.person_colors[person_name] = COLORS[self.color_idx % len(COLORS)]
            self.color_idx += 1
        return self.person_colors[person_name]
    
    def _calculate_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def _associate_face(self, bbox, frame_num):
        best_face_id = None
        best_iou = 0.0
        for face_id, tracker in self.face_trackers.items():
            if frame_num - tracker['last_seen'] > 30: continue
            iou = self._calculate_iou(bbox, tracker['last_bbox'])
            if iou > best_iou:
                best_iou = iou
                best_face_id = face_id
        if best_iou >= self.face_association_threshold:
            self.face_trackers[best_face_id]['last_bbox'] = bbox
            self.face_trackers[best_face_id]['last_seen'] = frame_num
            return best_face_id
        new_id = self.next_face_id
        self.next_face_id += 1
        self.face_trackers[new_id] = {
            'buffer': deque(maxlen=self.temporal_buffer_size),
            'last_bbox': bbox,
            'last_seen': frame_num
        }
        return new_id
    
    def _update_embedding_buffer(self, face_id, embedding):
        if face_id in self.face_trackers:
            self.face_trackers[face_id]['buffer'].append(embedding)
    
    def _get_smoothed_embedding(self, face_id):
        if face_id not in self.face_trackers: return None
        buffer = self.face_trackers[face_id]['buffer']
        if not buffer: return None
        embeddings = list(buffer)
        weights = np.exp(np.linspace(-1, 0, len(embeddings)))
        weights /= weights.sum()
        smoothed = np.zeros_like(embeddings[0])
        for i, emb in enumerate(embeddings):
            smoothed += weights[i] * emb
        return smoothed / np.linalg.norm(smoothed)
    
    def _cleanup_stale_trackers(self, current_frame, max_age=60):
        stale = [fid for fid, t in self.face_trackers.items() if current_frame - t['last_seen'] > max_age]
        for fid in stale: del self.face_trackers[fid]
    
    def extract_embedding(self, face, img):
        M = cv2.estimateAffinePartial2D(face.kps, Config.REF_LANDMARKS, method=cv2.RANSAC)[0]
        if M is None: return None
        aligned = cv2.warpAffine(img, M, Config.ALIGNED_FACE_SIZE)
        embedding = self.recognizer.get_feat(aligned)
        if embedding is None: return None
        embedding = embedding.flatten()
        return embedding / np.linalg.norm(embedding)
    
    def match_face(self, embedding):
        best_match = None
        best_score = -1.0
        
        for person_name, person_data in self.database.items():
            # Support both formats: templates or single embedding / list
            templates = []
            if isinstance(person_data, list):
                 templates = person_data
            elif isinstance(person_data, dict):
                templates = person_data.get('templates', [])
                if not templates and 'embedding' in person_data:
                    templates = [person_data['embedding']]
                if not templates and 'all_embeddings' in person_data:
                    templates = person_data['all_embeddings']
            else:
                templates = [person_data] if hasattr(person_data, 'shape') else []
            
            # Compare with each template
            for template in templates:
                template = np.array(template).flatten()
                similarity = np.dot(embedding, template)
                if similarity > best_score:
                    best_score = similarity
                    best_match = person_name
        
        if best_score >= self.threshold:
            return best_match, best_score
        else:
            return "Unknown", best_score
    
    def draw_results(self, frame, face, name, score):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        color = self.get_color(name) if name != "Unknown" else (128, 128, 128)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({score:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-20), (x1+w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        for kp in face.kps: cv2.circle(frame, (int(kp[0]), int(kp[1])), 2, color, -1)

    def process_stream(self, video_path):
        stride = Config.VIDEO_FRAME_STRIDE
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        last_results = []
        self.face_trackers = {}
        self.next_face_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            display_frame = frame.copy()
            if frame_count % stride == 1:
                faces = self.detector.get(frame)
                processed_count += 1
                if frame_count % 30 == 0: self._cleanup_stale_trackers(frame_count)
                current_results = []
                for face in faces:
                    q_res = self.quality_scorer.calculate(det_score=face.det_score, landmarks=face.kps)
                    if q_res['quality_score'] < self.min_quality_threshold: continue
                    embedding = self.extract_embedding(face, frame)
                    if embedding is None: continue
                    face_id = self._associate_face(face.bbox.astype(float).tolist(), frame_count)
                    self._update_embedding_buffer(face_id, embedding)
                    smoothed = self._get_smoothed_embedding(face_id)
                    name, score = self.match_face(smoothed if smoothed is not None else embedding)
                    current_results.append((face, name, score))
                    self.stats[name] += 1
                last_results = current_results
            else:
                current_results = last_results
            for res in current_results: self.draw_results(display_frame, *res)
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
