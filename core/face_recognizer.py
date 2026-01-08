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
from django.conf import settings

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
    
    def __init__(self, threshold=None, temporal_buffer_size=20, target_identities=None, min_face_size=None, min_blur=None):
        """
        Initialize recognizer.
        """
        self.threshold = threshold or Config.RECOGNITION_THRESHOLD
        self.temporal_buffer_size = temporal_buffer_size
        self.target_identities = target_identities
        
        # Dynamic Quality Settings
        self.min_face_size = min_face_size if min_face_size is not None else Config.MIN_FACE_SIZE
        self.min_blur = min_blur if min_blur is not None else Config.MIN_BLUR_SCORE
        
        print("\n" + "="*60)
        print("🎬 VIDEO FACE RECOGNIZER (Pickle Mode)")
        print(f"⚙️  Settings: Blur>{self.min_blur}, Size>{self.min_face_size}px")
        if self.target_identities:
            print(f"🎯 Context Active: Restricting search to {len(self.target_identities)} cast members.")
        print("="*60)
        
        # Load embeddings database from 'embeddings_all.pkl' (Rich) or 'avg_embeddings.pkl' (Simple)
        candidates = [
            'embeddings_all.pkl',
            os.path.join(Config.BASE_DIR, 'embeddings_all.pkl'),
            r'c:\Users\hamza\OneDrive\Belgeler\GitHub\video_face_embedding_ui\embeddings_all.pkl',
            'avg_embeddings.pkl',
            os.path.join(Config.BASE_DIR, 'avg_embeddings.pkl')
        ]
        
        self.database = {}
        loaded = False
        
        for path in candidates:
            if os.path.exists(path):
                print(f"📂 Loading database from: {path}")
                try:
                    with open(path, 'rb') as f:
                        raw_db = pickle.load(f)
                        
                    # Parse database (Handle both rich dict and simple dict)
                    self.database = {}
                    count_templates = 0
                    
                    # Filter keys if context is active
                    filtered_keys = raw_db.keys()
                    if self.target_identities:
                        filtered_keys = [k for k in raw_db.keys() if k in self.target_identities]
                        print(f"   🎯 Context Filter: {len(filtered_keys)} / {len(raw_db)} identities")
                        
                    for name in filtered_keys:
                        val = raw_db[name]
                        templates = []
                        
                        if isinstance(val, dict):
                            # Rich structure from embeddings_all.pkl
                            if 'templates' in val and len(val['templates']) > 0:
                                templates = val['templates']
                            elif 'all_embeddings' in val and len(val['all_embeddings']) > 0:
                                # Fallback to 10 random samples if no templates but raw embeddings exist
                                embeddings_list = val['all_embeddings']
                                step = max(1, len(embeddings_list) // 10)
                                templates = embeddings_list[::step][:10]
                            elif 'embedding' in val:
                                templates = [val['embedding']]
                        else:
                            # Simple structure from avg_embeddings.pkl (numpy array or list)
                            templates = [val]
                            
                        # Normalize and Validate
                        valid_templates = []
                        for t in templates:
                            t = np.array(t).flatten()
                            norm = np.linalg.norm(t)
                            if norm > 0:
                                valid_templates.append(t / norm)
                                
                        if valid_templates:
                            self.database[name] = valid_templates
                            count_templates += len(valid_templates)

                    print(f"   ✓ Loaded {len(self.database)} persons with {count_templates} total templates.")
                    
                    # ---------------------------------------------------------
                    # 🚀 OPTIMIZATION: Build Numpy Feature Matrix
                    # ---------------------------------------------------------
                    print("   ⚙️  Building Vectorized Feature Matrix...")
                    t_start = time.time()
                    
                    self.names_list = []
                    feats_list = []
                    
                    for name, templates in self.database.items():
                        for feat in templates:
                            self.names_list.append(name)
                            feats_list.append(feat)
                            
                    if feats_list:
                        self.feature_matrix = np.array(feats_list, dtype=np.float32)
                        # Transpose for (N, 512) dot (512,) -> (N,)
                        # Actually keeping it (N, 512) allows np.dot(matrix, vector)
                    else:
                        self.feature_matrix = None
                        
                    print(f"   🚀 Matrix Built: {self.feature_matrix.shape if self.feature_matrix is not None else 'Empty'} in {time.time()-t_start:.3f}s")
                    
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
    
    def _associate_face(self, bbox, frame_num, embedding):
        """
        Associate detection with trackers using DeepSORT-lite logic.
        (Appearance + IoU)
        """
        best_face_id = None
        best_score = -1.0 # Combined score
        
        # Hyperparameters (Relaxed for stability)
        # Shift weight to IoU because position is reliable in short-term even if lookalike fails
        iou_weight = 0.6      # Increased from 0.3
        app_weight = 0.4      # Decreased from 0.7
        min_iou_gate = 0.1    # Keep low, must overlap
        min_cosine_gate = 0.3 # Lowered from 0.5 (Allow more appearance variation)
        
        for face_id, tracker in self.face_trackers.items():
            if frame_num - tracker['last_seen'] > 30: continue
            
            # 1. IoU Score
            iou = self._calculate_iou(bbox, tracker['last_bbox'])
            
            # 2. Appearance Score (Cosine Similarity)
            # Use the smoothed embedding of the tracker
            tracker_emb = self._get_smoothed_embedding(face_id)
            cosine_score = 0.0
            if tracker_emb is not None:
                cosine_score = np.dot(embedding, tracker_emb)
            else:
                # Fallback if no history yet (shouldn't happen often)
                cosine_score = 0.5 
            
            # 3. Gating (Filters)
            if iou < min_iou_gate or cosine_score < min_cosine_gate:
                continue
                
            # 4. Combined Score
            combined_score = (iou * iou_weight) + (cosine_score * app_weight)
            
            if combined_score > best_score:
                best_score = combined_score
                best_face_id = face_id
                
        # Threshold for assignment
        if best_score > 0.3: # Lowered from 0.4 for better stickiness
            self.face_trackers[best_face_id]['last_bbox'] = bbox
            self.face_trackers[best_face_id]['last_seen'] = frame_num
            return best_face_id
            
        # New Tracker
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
        # Exponential weighted moving average with sharper decay
        # -1 to 0 gives e^-1 (0.36) weight to oldest
        # -3 to 0 gives e^-3 (0.05) weight to oldest (Forgets bad frames faster)
        weights = np.exp(np.linspace(-3, 0, len(embeddings)))
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
        """
        Match embedding against database (Multi-Template).
        If no match, cluster the unknown face to give it a temporary ID.
        """
        best_match = None
        best_score = -1.0
        
        # 1. Known Matching (Vectorized Multi-Template)
        # ---------------------------------------------
        if self.feature_matrix is not None:
            # Score = Matrix (N, 512) x Vector (512,) -> (N,)
            scores = np.dot(self.feature_matrix, embedding)
            
            # Find best match
            best_idx = np.argmax(scores)
            max_score = scores[best_idx]
            
            if max_score > best_score:
                best_score = max_score
                best_match = self.names_list[best_idx]
        
        if best_score >= self.threshold:
            return best_match, best_score
            
        # 2. Unknown Clustering (Dynamic Guest IDs)
        # ---------------------------------------------
        # Check against existing "Guest" clusters
        if not hasattr(self, 'unknown_clusters'):
            self.unknown_clusters = []  # List of {'id': int, 'embedding': array, 'count': int}

        best_guest_id = None
        best_guest_score = -1.0
        guest_threshold = 0.65  # High enough to be sure it's the same stranger

        for cluster in self.unknown_clusters:
            similarity = np.dot(embedding, cluster['embedding'])
            if similarity > best_guest_score:
                best_guest_score = similarity
                best_guest_id = cluster['id']

        if best_guest_score > guest_threshold:
            # Update cluster centroid (Moving Average)
            for cluster in self.unknown_clusters:
                if cluster['id'] == best_guest_id:
                    # Simple weighted average
                    cluster['embedding'] = 0.9 * cluster['embedding'] + 0.1 * embedding
                    cluster['embedding'] /= np.linalg.norm(cluster['embedding'])
                    cluster['count'] += 1
                    return f"Misafir-{best_guest_id}", best_guest_score
        else:
            # Create new Guest ID
            new_guest_id = len(self.unknown_clusters) + 1
            self.unknown_clusters.append({
                'id': new_guest_id,
                'embedding': embedding,
                'count': 1
            })
            return f"Misafir-{new_guest_id}", 0.0 # Score 0 because it's new

        return "Unknown", best_score
        
    def find_nearest_neighbors(self, embedding, k=5):
        """
        Find top K matches for the embedding.
        Returns list of tuples: (name, score)
        """
        if self.feature_matrix is None:
            return []
            
        # Check against all knowns
        scores = np.dot(self.feature_matrix, embedding)
        
        # Get top K indices
        # np.argsort returns indices that sort the array. We want descending.
        # This is full sort, can be slow for huge DBs. argpartition is faster but not sorted.
        # For <100k, sort is fine.
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        seen_names = set()
        
        for idx in top_indices:
            name = self.names_list[idx]
            score = float(scores[idx])
            
            # Since we have multiple templates per person, we might get multiple hits for same person.
            # We usually only want the best score per person.
            if name not in seen_names:
                results.append({'name': name, 'score': score})
                seen_names.add(name)
                
            if len(results) >= k:
                break
                
        return results
    
    def draw_results(self, frame, face, name, score):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Determine color: Gray for Unknown/Misafir, Custom for others
        is_unknown = name == "Unknown" or name.startswith("Misafir")
        color = (128, 128, 128) if is_unknown else self.get_color(name)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({score:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-20), (x1+w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        for kp in face.kps: cv2.circle(frame, (int(kp[0]), int(kp[1])), 2, color, -1)

    def process_stream(self, video_path, session_id=None):
        from django.core.cache import cache # Lazy import to keep class portable if needed
        
        stride = Config.VIDEO_FRAME_STRIDE
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        last_results = []
        self.face_trackers = {}
        self.next_face_id = 0
        
        # Initial status
        if session_id:
            cache.set(f"rec_stats_{session_id}", {
                'status': 'running', 
                'progress': 0, 
                'stats': {}
            }, timeout=300)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            display_frame = frame.copy()
            
            # --- Detection & Recognition ---
            if frame_count % stride == 1:
                faces = self.detector.get(frame)
                processed_count += 1
                if frame_count % 30 == 0: self._cleanup_stale_trackers(frame_count)
                current_results = []
                for face in faces:
                    # Manual Quality Filtering (User Controls)
                    # 1. Size Check
                    bbox = face.bbox.astype(int)
                    face_h = bbox[3] - bbox[1]
                    if face_h < self.min_face_size:
                         continue
                         
                    # 2. Blur Check Matches Config
                    # We use the quality scorer which internally calculates blur
                    # But if we want to reject specifically on blur before complex calc:
                    
                    # 3. HUMAN GEOMETRY CHECK (Anti-Animal)
                    if Config.ENABLE_LANDMARK_CHECK:
                        kps = face.kps
                        left_eye = kps[0]
                        right_eye = kps[1]
                        eye_dist = np.linalg.norm(right_eye - left_eye)
                        face_width = bbox[2] - bbox[0]
                        
                        if face_width > 0:
                            eye_ratio = eye_dist / face_width
                            # Use Config values
                            min_ratio = getattr(Config, 'MIN_EYE_DISTANCE_RATIO', 0.20)
                            max_ratio = getattr(Config, 'MAX_EYE_DISTANCE_RATIO', 0.65)
                            
                            if eye_ratio < min_ratio or eye_ratio > max_ratio:
                                continue # Skip non-human face

                    # 4. Global Quality Score (Includes Blur, Brightness, Pose)
                    q_res = self.quality_scorer.calculate(det_score=face.det_score, landmarks=face.kps)
                    
                    # Specific Blur Check from Scorer results if min_blur is set
                    # Note: Scorer returns normalized 0-1 blur score, but min_blur user input is 0-500 variance.
                    # So we should stick to raw variance for consistency with UI slider or convert.
                    # Currently UI slider sends Variance (e.g. 100). Scorer uses internal Normalization.
                    # Let's trust the unified score OR explicit variance check if critical.
                    # For now, let's keep the explicit variance check if user provided it via UI slider params.
                    if self.min_blur > 0:
                         x1, y1, x2, y2 = bbox
                         h, w, _ = frame.shape
                         x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                         if x2>x1 and y2>y1:
                             face_crop = frame[y1:y2, x1:x2]
                             gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                             blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
                             if blur_val < self.min_blur:
                                 continue

                    if q_res['quality_score'] < self.min_quality_threshold: continue
                    embedding = self.extract_embedding(face, frame)
                    if embedding is None: continue
                    # Pass embedding for DeepSORT-lite logic
                    face_id = self._associate_face(face.bbox.astype(float).tolist(), frame_count, embedding)
                    self._update_embedding_buffer(face_id, embedding)
                    smoothed = self._get_smoothed_embedding(face_id)
                    name, score = self.match_face(smoothed if smoothed is not None else embedding)
                    current_results.append((face, name, score))
                    
                    if name.startswith("Misafir") and session_id:
                        # Save Guest Snapshot & Embedding
                        guest_dir = os.path.join(settings.MEDIA_ROOT, 'temp_faces', session_id)
                        os.makedirs(guest_dir, exist_ok=True)
                        guest_path = os.path.join(guest_dir, f"{name}.jpg")
                        guest_npy_path = os.path.join(guest_dir, f"{name}.npy")
                        
                        # Save if either missing (Robustness for re-runs)
                        if not os.path.exists(guest_path) or not os.path.exists(guest_npy_path):
                            # Save Embedding
                            np.save(guest_npy_path, embedding)

                            # Crop Face (only if missing)
                            if not os.path.exists(guest_path):
                                bbox = face.bbox.astype(int)
                                x1, y1, x2, y2 = bbox
                                h, w, _ = frame.shape
                                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                                if x2>x1 and y2>y1:
                                    face_img = frame[y1:y2, x1:x2]
                                    cv2.imwrite(guest_path, face_img)

                    # Update stats
                    self.stats[name] += 1
                    
                last_results = current_results
                
                # Update Cache frequently (every processed frame)
                if session_id:
                    progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
                    
                    # Debug print to ensure stats are collected
                    if processed_count % 10 == 0:
                        print(f"📊 Stats Sync [Session {session_id}]: {len(self.stats)} actors found.")
                    
                    # Sort stats by count desc
                    sorted_stats = dict(sorted(self.stats.items(), key=lambda item: item[1], reverse=True))
                    
                    cache.set(f"rec_stats_{session_id}", {
                        'status': 'running', 
                        'progress': progress, 
                        'stats': sorted_stats,
                        'fps': frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                    }, timeout=60)
                    
            else:
                current_results = last_results
            
            # Draw
            for res in current_results: self.draw_results(display_frame, *res)
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        cap.release()
        
        # Final status update
        if session_id:
            sorted_stats = dict(sorted(self.stats.items(), key=lambda item: item[1], reverse=True))
            cache.set(f"rec_stats_{session_id}", {
                'status': 'completed', 
                'progress': 100, 
                'stats': sorted_stats
            }, timeout=300)
