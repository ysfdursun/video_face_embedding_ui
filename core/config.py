"""
Centralized Configuration (from Demo + Django Integration)
"""

import os
import numpy as np


class Config:
    """Pipeline configuration container"""
    
    # ==========================================
    # PATHS
    # ==========================================
    # Will be set from Django settings
    BASE_DIR = None
    OUTPUT_DIR = None
    
    # ==========================================
    # MODEL SETTINGS
    # ==========================================
    
    # Detection model (buffalo_sc = SCRFD 2.5G)
    DETECTION_MODEL = "buffalo_sc"
    DETECTION_SIZE = (640, 640)
    DETECTION_THRESHOLD = 0.65  # Stricter than Django default (0.5)
    
    # Recognition model (buffalo_l = w600k_r50)
    RECOGNITION_MODEL = "buffalo_l"
    RECOGNITION_MODEL_FILE = "w600k_r50.onnx"
    
    # Execution providers
    PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # ==========================================
    # ALIGNMENT
    # ==========================================
    
    # ArcFace reference landmarks (112x112)
    REF_LANDMARKS = np.array([
        [38.2946, 51.6963],   # Left eye
        [73.5318, 51.5014],   # Right eye
        [56.0252, 71.7366],   # Nose tip
        [41.5493, 92.3655],   # Left mouth
        [70.7299, 92.2041]    # Right mouth
    ], dtype=np.float32)
    
    ALIGNED_FACE_SIZE = (112, 112)
    
    # ==========================================
    # QUALITY THRESHOLDS
    # ==========================================
    
    # Blur - minimum sharpness (Laplacian variance)
    MIN_BLUR_SCORE = 80.0  # Stricter than Django (30.0)
    
    # Brightness range
    MIN_BRIGHTNESS = 30
    MAX_BRIGHTNESS = 230
    
    # Face size
    MIN_FACE_SIZE = 50  # More lenient than Django (80)
    
    # Pose filtering
    MAX_POSE_RATIO = 2.0  # Filter extreme poses
    
    # Landmark quality
    ENABLE_LANDMARK_CHECK = True
    MIN_EYE_DISTANCE_RATIO = 0.2
    MAX_EYE_DISTANCE_RATIO = 0.7
    
    # ==========================================
    # OUTLIER DETECTION
    # ==========================================
    
    OUTLIER_SIMILARITY_THRESHOLD = 0.4
    MIN_EMBEDDINGS_FOR_OUTLIER = 3
    
    # ==========================================
    # VIDEO PROCESSING
    # ==========================================
    
    VIDEO_FRAME_STRIDE = 3  # Process every N frames (Django: 10)
    GROUPING_THRESHOLD = 0.60  # Stricter than Django (0.45)
    RECOGNITION_THRESHOLD = 0.35
    
    # ==========================================
    # ENHANCEMENT
    # ==========================================
    
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    LOW_BRIGHTNESS_THRESHOLD = 40
    GAMMA_CORRECTION = 1.5
    
    # ==========================================
    # DJANGO INTEGRATION
    # ==========================================
    
    @classmethod
    def sync_from_django_settings(cls):
        """Load and override settings from Django"""
        try:
            from django.conf import settings as django_settings
            
            # Set paths from Django
            cls.BASE_DIR = django_settings.BASE_DIR
            cls.OUTPUT_DIR = os.path.join(django_settings.MEDIA_ROOT, "demo_outputs")
            
            # Try to load from FaceRecognitionSettings model
            try:
                from core.models import FaceRecognitionSettings
                db_settings = FaceRecognitionSettings.get_settings()
                
                # Override thresholds if DB values exist
                if db_settings.detection_threshold:
                    cls.DETECTION_THRESHOLD = db_settings.detection_threshold
                if db_settings.grouping_threshold:
                    cls.GROUPING_THRESHOLD = db_settings.grouping_threshold
                if db_settings.min_face_size:
                    cls.MIN_FACE_SIZE = db_settings.min_face_size
                if db_settings.frame_skip_extract:
                    cls.VIDEO_FRAME_STRIDE = db_settings.frame_skip_extract
                
                # Map quality_threshold to blur equivalent
                if db_settings.quality_threshold:
                    # quality_threshold is 0-1, MIN_BLUR_SCORE is raw variance
                    # Keep MIN_BLUR_SCORE from demo (80.0) as it's better
                    pass
                
                print(f"✓ Config synced from Django DB")
            except Exception as e:
                print(f"⚠ Using demo config defaults: {e}")
        
        except Exception as e:
            print(f"⚠ Django settings not available: {e}")
    
    @classmethod
    def get_insightface_root(cls):
        """Get InsightFace models root directory"""
        return os.path.join(os.path.expanduser("~"), ".insightface")
    
    @classmethod
    def get_recognition_model_path(cls):
        """Get full path to recognition model"""
        return os.path.join(
            cls.get_insightface_root(),
            "models",
            cls.RECOGNITION_MODEL,
            cls.RECOGNITION_MODEL_FILE
        )
