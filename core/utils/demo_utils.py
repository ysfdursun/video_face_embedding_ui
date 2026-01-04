"""
Shared Utilities
================
Common helper classes and functions for face processing.
"""

import cv2
import numpy as np
from core.config import Config


# ==========================================
# IMAGE ENHANCEMENT
# ==========================================

class AutoEnhancer:
    """
    Automatic image enhancement for low-light images.
    Uses CLAHE + Gamma correction.
    """
    
    def __init__(self, clip_limit=None, tile_grid_size=None):
        clip_limit = clip_limit or Config.CLAHE_CLIP_LIMIT
        tile_grid_size = tile_grid_size or Config.CLAHE_TILE_SIZE
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, 
            tileGridSize=tile_grid_size
        )
    
    def adjust_gamma(self, image, gamma=1.0):
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 
            for i in np.arange(256)
        ]).astype("uint8")
        return cv2.LUT(image, table)
    
    def enhance(self, img):
        """
        Enhance image if needed.
        
        Args:
            img: BGR image
            
        Returns:
            tuple: (enhanced_img, brightness, technique_used)
        """
        # Handle grayscale or BGRA
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Calculate brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        enhanced_img = img.copy()
        technique = "None"
        
        # Apply enhancement for low-light images
        if brightness < Config.LOW_BRIGHTNESS_THRESHOLD:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            enhanced_img = self.adjust_gamma(enhanced_img, gamma=Config.GAMMA_CORRECTION)
            technique = "CLAHE+Gamma"
        
        return enhanced_img, brightness, technique


# ==========================================
# POSE ANALYSIS
# ==========================================

class PoseAnalyzer:
    """
    Face pose estimation from 5-point landmarks.
    Classifies faces as Frontal, Tilted, or Profile.
    """
    
    @staticmethod
    def analyze(landmarks):
        """
        Analyze face pose from landmarks.
        
        Args:
            landmarks: 5-point facial landmarks array
            
        Returns:
            tuple: (pose_label, ratio_value)
        """
        if landmarks is None or len(landmarks) < 3:
            return "Unknown", 1.0
        
        # Calculate distances from eyes to nose
        dist_left_eye_nose = np.linalg.norm(landmarks[0] - landmarks[2])
        dist_right_eye_nose = np.linalg.norm(landmarks[1] - landmarks[2])
        
        if dist_left_eye_nose == 0 or dist_right_eye_nose == 0:
            return "Unknown", 0.0
        
        # Ratio indicates face rotation
        ratio = max(dist_left_eye_nose, dist_right_eye_nose) / \
                min(dist_left_eye_nose, dist_right_eye_nose)
        
        if ratio < 1.4:
            return "Frontal", ratio
        elif ratio < 2.0:
            return "Tilted", ratio
        else:
            return "Profile", ratio


# ==========================================
# EMBEDDING UTILITIES
# ==========================================

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def validate_embedding_norm(embedding, tolerance=0.01):
    """Validate and normalize embedding to unit length."""
    norm = np.linalg.norm(embedding)
    
    if norm == 0:
        return None
    
    if abs(norm - 1.0) > tolerance:
        return embedding / norm
    
    return embedding


def calculate_aspect_ratio(bbox):
    """Calculate width/height ratio from bounding box."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width / height if height > 0 else 1.0


def detect_outliers(embeddings, files, threshold=None):
    """Detect outliers based on self-similarity."""
    threshold = threshold or Config.OUTLIER_SIMILARITY_THRESHOLD
    
    if len(embeddings) < Config.MIN_EMBEDDINGS_FOR_OUTLIER:
        return embeddings, files, []
    
    embeddings_arr = np.array(embeddings)
    n = len(embeddings)
    
    # Calculate average similarity for each embedding
    avg_similarities = []
    for i in range(n):
        sims = [np.dot(embeddings_arr[i], embeddings_arr[j]) 
                for j in range(n) if i != j]
        avg_similarities.append(np.mean(sims))
    
    # Filter outliers
    clean_embeddings = []
    clean_files = []
    outlier_files = []
    
    for emb, file, avg_sim in zip(embeddings, files, avg_similarities):
        if avg_sim >= threshold:
            clean_embeddings.append(emb)
            clean_files.append(file)
        else:
            outlier_files.append((file, avg_sim))
    
    return clean_embeddings, clean_files, outlier_files


# ==========================================
# IMAGE I/O
# ==========================================

def read_image_safe(image_path):
    """Read image with Unicode path support."""
    try:
        with open(image_path, "rb") as f:
            bytes_data = bytearray(f.read())
        img = cv2.imdecode(np.asarray(bytes_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


def save_image(image, output_path, quality=95):
    """Save image with specified quality."""
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
