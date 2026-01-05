# -*- coding: utf-8 -*-
"""
Face Processor Module - Robust Version
======================================
Integrates "Standard" extraction logic:
- AutoEnhancement (CLAHE)
- ArcFace Alignment
- Strict Quality Checks
- Face Grouping

Adapted for Django Streaming.
"""

import os
import cv2
import numpy as np
import time
import json
from datetime import datetime
from collections import defaultdict

from django.conf import settings
from django.core.cache import cache

from core.config import Config
from core.model_loader import load_detector, load_recognizer
from core.utils.demo_utils import AutoEnhancer, PoseAnalyzer, save_image
from core.utils.file_utils import get_safe_filename
from core.models import FaceGroup as DBFaceGroup, Movie

class FaceGroupHelper:
    """Represents a group of similar faces (memory representation)."""
    
    def __init__(self, group_id):
        self.group_id_int = group_id
        self.faces = []           # List of face images
        self.embeddings = []      # List of embeddings
        self.metadata = []        # List of metadata dicts
        self.representative = None  # Best quality face index
        self.rep_embedding = None # Representative embedding (EMA)
        
        # Stats
        self.total_quality = 0.0
        self.quality_count = 0
        self.total_sim = 0.0
        self.sim_count = 0
    
    def add_face(self, face_img, embedding, metadata):
        """Add a face to this group."""
        self.faces.append(face_img)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        
        # Update EMA representative embedding
        if self.rep_embedding is None:
            self.rep_embedding = embedding
        else:
            self.rep_embedding = 0.9 * self.rep_embedding + 0.1 * embedding
            
        # Update representative face (Best Angle + Quality)
        if self.representative is None:
            self.representative = len(self.faces) - 1
        else:
            curr_meta = self.metadata[self.representative]
            new_meta = metadata
            
            # 1. Prefer Frontal Faces
            curr_is_frontal = curr_meta.get('pose') == 'Frontal'
            new_is_frontal = new_meta.get('pose') == 'Frontal'
            
            if new_is_frontal and not curr_is_frontal:
                 self.representative = len(self.faces) - 1
            elif curr_is_frontal and not new_is_frontal:
                 pass # Keep current
            else:
                 # 2. Tie-break with Blur Score (Sharpness)
                 current_blur = curr_meta.get('blur_score', 0)
                 new_blur = new_meta.get('blur_score', 0)
                 if new_blur > current_blur:
                     self.representative = len(self.faces) - 1
                
        # Update stats
        quality = metadata.get('quality_score', 0)
        self.total_quality += quality
        self.quality_count += 1
        
        if 'sim_score' in metadata:
            self.total_sim += metadata['sim_score']
            self.sim_count += 1
            
    def get_centroid(self):
        """Get average embedding of the group."""
        if not self.embeddings:
            return None
        centroid = np.mean(self.embeddings, axis=0)
        return centroid / np.linalg.norm(centroid)


class VideoFaceExtractor:
    """
    Robust Video Face Extractor adapted for Streaming.
    """
    
    def __init__(self, output_dir, grouping_threshold=0.6, 
                 min_group_size=2, enable_quality_filter=True):
        self.output_dir = output_dir
        self.grouping_threshold = grouping_threshold
        self.min_group_size = min_group_size
        self.enable_quality_filter = enable_quality_filter
        
        # Create output directories
        self.groups_dir = os.path.join(output_dir, "groups") # or separate folders per group
        # For Django compatibility, we might want "celebrity_XXX" folders directly in output_dir
        # Let's stick to the user's snippet logic:
        # snippet: out_dir/celebrity_001/img_001.jpg
        self.base_output_dir = output_dir
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # Load models
        self.detector = load_detector()
        self.recognizer = load_recognizer()
        self.enhancer = AutoEnhancer()
        self.pose_analyzer = PoseAnalyzer()
        
        self.groups = [] # List of FaceGroupHelper
        self.stats = defaultdict(int)

    def check_quality(self, face, aligned_face, brightness):
        """Check if face passes quality thresholds."""
        # Detection score check
        if face.det_score < Config.DETECTION_THRESHOLD:
            return False, f"low_det_score ({face.det_score:.3f})", 0.0
        
        # Face size check
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width < Config.MIN_FACE_SIZE or height < Config.MIN_FACE_SIZE:
            return False, f"too_small ({width:.0f}x{height:.0f})", 0.0
            
        # Blur check (Laplacian)
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < Config.MIN_BLUR_SCORE:
            self.stats['reject_blur'] += 1
            return False, f"blur ({blur_score:.1f})", 0.0
            
        # Brightness check
        if brightness < Config.MIN_BRIGHTNESS:
            return False, f"too_dark ({brightness:.1f})", 0.0
        if brightness > Config.MAX_BRIGHTNESS:
            return False, f"too_bright ({brightness:.1f})", 0.0
            
        # Calculate composite score
        quality_score = (face.det_score * 30) + (min(blur_score, 500) / 500 * 30) + 20
        return True, "passed", quality_score

    def align_face(self, img, landmarks):
        """Align face using ArcFace transformation."""
        if landmarks is None:
            return None
        M = cv2.estimateAffinePartial2D(
            landmarks, Config.REF_LANDMARKS, method=cv2.RANSAC
        )[0]
        if M is None:
            return None
        aligned = cv2.warpAffine(
            img, M, Config.ALIGNED_FACE_SIZE, borderMode=cv2.BORDER_REPLICATE
        )
        return aligned

    def extract_embedding(self, aligned_face):
        """Extract normalized embedding."""
        embedding = self.recognizer.get_feat(aligned_face)
        if embedding is None:
            return None
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def find_matching_group(self, embedding):
        """Find matching group."""
        best_group_idx = None
        best_similarity = 0.0
        
        for i, group in enumerate(self.groups):
            # Compare with representative embedding (EMA)
            if group.rep_embedding is not None:
                similarity = np.dot(embedding, group.rep_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_group_idx = i
                    
        if best_similarity >= self.grouping_threshold:
            return best_group_idx, best_similarity
        return None, best_similarity

    def save_results(self, movie_obj):
        """Save results to filesystem and DB."""
        print(f"\n💾 Saving {len(self.groups)} groups...")
        
        for i, group in enumerate(self.groups):
            # Filter small groups
            if len(group.faces) < self.min_group_size:
                self.stats['groups_too_small'] += 1
                continue
                
            group_name = f"celebrity_{i:03d}"
            group_dir = os.path.join(self.base_output_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)
            
            # Save faces
            for j, face_img in enumerate(group.faces):
                save_image(face_img, os.path.join(group_dir, f"img_{j:04d}.jpg"))
            
            # Save to DB
            try:
                avg_conf = group.total_sim / group.sim_count if group.sim_count > 0 else 1.0
                avg_qual = group.total_quality / group.quality_count if group.quality_count > 0 else 0.0
                
                fg, _ = DBFaceGroup.objects.get_or_create(
                    movie=movie_obj,
                    group_id=group_name
                )
                fg.face_count = len(group.faces)
                fg.total_faces = len(group.faces)
                fg.avg_confidence = avg_conf
                fg.avg_quality = avg_qual
                
                if group.rep_embedding is not None:
                    fg.set_representative_embedding(group.rep_embedding)
                
                fg.update_risk_level()
                self.stats['groups_saved'] += 1
                
            except Exception as e:
                print(f"Error saving group {group_name}: {e}")

    def process_stream(self, video_path, cache_key):
        """Yield MJPEG frames while processing."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = Config.VIDEO_FRAME_STRIDE
        
        frame_idx = 0
        pbar_last_update = 0
        
        # FPS display vars
        fps_start = time.time()
        fps_counter = 0
        current_fps = 0.0
        
        # Store for display on skipped frames
        self.last_display_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            display_frame = frame.copy()
            
            # FPS Calculation
            fps_counter += 1
            if time.time() - fps_start > 1.0:
                current_fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()

            # Process every N frames
            if frame_idx % stride == 0:
                # 1. Enhance
                enhanced, brightness, technique = self.enhancer.enhance(frame)
                if technique != "None":
                    self.stats['enhanced'] += 1
                    
                # 2. Detect
                faces = self.detector.get(enhanced)
                
                current_faces_info = [] # for visualization
                
                if faces:
                    for face in faces:
                        self.stats['detected'] += 1
                        
                        # Align
                        aligned = self.align_face(enhanced, face.kps)
                        if aligned is None:
                            self.stats['align_error'] += 1
                            continue
                            
                        # Quality Check
                        if self.enable_quality_filter:
                            passed, reason, quality_score = self.check_quality(face, aligned, brightness)
                            if not passed:
                                self.stats['quality_fail'] += 1
                                # Draw rejection box
                                bbox = face.bbox.astype(int)
                                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                                continue
                        else:
                             quality_score = 0.0
                        
                        self.stats['quality_pass'] += 1
                        
                        # Embed
                        embedding = self.extract_embedding(aligned)
                        if embedding is None:
                            self.stats['embed_error'] += 1
                            continue
                            
                        # Group
                        group_idx, sim_score = self.find_matching_group(embedding)
                        
                        if group_idx is None:
                            # New group
                            group_idx = len(self.groups)
                            self.groups.append(FaceGroupHelper(group_idx))
                            self.stats['groups_new'] += 1
                            sim_score = 1.0
                        
                        # Calculate blur score
                        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        
                        # Pose Analysis
                        pose, ratio = self.pose_analyzer.analyze(face.kps)
                        self.stats[f'pose_{pose.lower()}'] += 1
                        
                        metadata = {
                            'frame_idx': frame_idx,
                            'quality_score': quality_score,
                            'sim_score': sim_score,
                            'blur_score': float(blur_score),
                            'pose': pose,
                            'pose_ratio': float(ratio)
                        }
                        
                        self.groups[group_idx].add_face(aligned, embedding, metadata)
                        self.stats['extracted'] += 1
                        
                        # Add to visualization list
                        current_faces_info.append((face.bbox.astype(int), group_idx, f"{pose[0]} Q:{quality_score:.0f}"))

                # Update last display info
                self.last_display_info = current_faces_info if faces else []

            # Reuse last detection info for visualization (to avoid flicker)
            if hasattr(self, 'last_display_info'):
                for bbox, group_id, quality in self.last_display_info:
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"G{group_id} {quality}", 
                               (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw Overlay (Progress + FPS)
            progress = (frame_idx / total_frames) * 100
            
            # Black background box
            cv2.rectangle(display_frame, (5, 5), (450, 60), (0, 0, 0), -1)
            
            info_text = f"FPS: {current_fps:.1f} | Prog: {progress:.1f}%"
            stats_text = f"Groups: {len(self.groups)} | Extracted: {self.stats['extracted']}"
            
            cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(display_frame, stats_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Yield frame
            is_success, buffer = cv2.imencode(".jpg", display_frame)
            if is_success:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                       
        cap.release()
        
        # Verify if successful
        if self.stats['extracted'] > 0 or len(self.groups) > 0:
            cache.set(cache_key, "completed", timeout=3600)
        else:
            cache.set(cache_key, "completed", timeout=3600) # Still complete even if empty

def process_and_extract_faces_stream(video_path, movie_title, group_faces=True):
    """
    Main entry point for Django View.
    Wraps the Robust VideoFaceExtractor.
    """
    safe_title = get_safe_filename(movie_title)
    cache_key = f"processing_status_{safe_title}"
    cache.set(cache_key, "running", timeout=3600)
    
    # 1. Setup Models & DB
    # Ensure movie exists
    movie_obj, _ = Movie.objects.get_or_create(title=safe_title)
    
    # Output dir
    output_dir = os.path.join(settings.MEDIA_ROOT, 'grouped_faces', safe_title)
    if os.path.exists(output_dir):
        import shutil
        try:
            shutil.rmtree(output_dir)
            DBFaceGroup.objects.filter(movie=movie_obj).delete()
        except:
            pass
            
    # 2. Synch Config
    # Config.sync_from_django_settings()
    
    # 3. Instantiate Extractor
    extractor = VideoFaceExtractor(
        output_dir=output_dir,
        grouping_threshold=Config.GROUPING_THRESHOLD,
        min_group_size=2,
        enable_quality_filter=True
    )
    
    try:
        # 4. Stream Loop
        yield from extractor.process_stream(video_path, cache_key)
        
        # 5. Save Final Results
        extractor.save_results(movie_obj)
        
        print(f"✅ Processing complete for {movie_title}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        cache.set(cache_key, "error", timeout=3600)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
