"""
Centralized Configuration
=========================
All configurable parameters for the face recognition pipeline.
"""

import os
import numpy as np


class Config:
    """Pipeline configuration container"""
    
    # ==========================================
    # PATHS
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Default dataset (can be overridden)
    # Using a safe relative path or environment variable is better in production, 
    # but keeping user's preference here.
    DATASET_DIR = r"C:\Users\hamza\OneDrive\Belgeler\GitHub\celebrity-face-recognition-project\Merged_Celebrity_Actors"
    
    # Output subdirectories
    CROPPED_FACES_DIR = os.path.join(OUTPUT_DIR, "cropped_faces")
    EMBEDDINGS_DIR = os.path.join(OUTPUT_DIR, "embeddings")
    ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
    
    # ==========================================
    # MODEL SETTINGS
    # ==========================================
    
    # Detection model (buffalo_sc = SCRFD 2.5G)
    DETECTION_MODEL = "buffalo_sc"
    DETECTION_SIZE = (640, 640)
    DETECTION_THRESHOLD = 0.35
    
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
    MIN_BLUR_SCORE = 80.0         # Increased from 50 - filters blurry
    
    # Brightness range
    MIN_BRIGHTNESS = 30
    MAX_BRIGHTNESS = 230
    
    # Face size
    MIN_FACE_SIZE = 50            # Increased from 40 - filters tiny faces
    
    # Detection confidence
    DETECTION_THRESHOLD = 0.65    # Increased - filters false positives
    
    # Pose filtering
    MAX_POSE_RATIO = 2.0          # Filter extreme poses (profile views)
    
    # Landmark quality
    ENABLE_LANDMARK_CHECK = True   # Check landmark positions
    MIN_EYE_DISTANCE_RATIO = 0.2   # Min eye distance / face width
    MAX_EYE_DISTANCE_RATIO = 0.7   # Max eye distance / face width
    
    # ==========================================
    # OUTLIER DETECTION
    # ==========================================
    
    OUTLIER_SIMILARITY_THRESHOLD = 0.4
    MIN_EMBEDDINGS_FOR_OUTLIER = 3
    
    # ==========================================
    # VIDEO PROCESSING
    # ==========================================
    
    VIDEO_FRAME_STRIDE = 3        # Process every N frames
    RECOGNITION_THRESHOLD = 0.35  # Cosine similarity threshold
    GROUPING_THRESHOLD = 0.6      # For face grouping (cosine sim)
    DUPLICATE_THRESHOLD = 0.95    # For deduplication (cosine sim)
    MIN_QUALITY_SCORE = 0.3       # For recognition filtering
    
    # ==========================================
    # ENHANCEMENT
    # ==========================================
    
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    LOW_BRIGHTNESS_THRESHOLD = 40
    GAMMA_CORRECTION = 1.5
    
    # ==========================================
    # METHODS
    # ==========================================
    
    @classmethod
    def ensure_dirs(cls):
        """Create all output directories if they don't exist"""
        for dir_path in [cls.OUTPUT_DIR, cls.CROPPED_FACES_DIR, 
                         cls.EMBEDDINGS_DIR, cls.ANALYSIS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
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
