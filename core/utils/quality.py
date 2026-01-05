"""
Face Quality Score Module
=========================
Unified quality scoring for face images.

Combines multiple quality metrics into a single score:
- Blur score (sharpness)
- Brightness score
- Pose score
- Detection confidence

Usage:
    from core.utils.quality import FaceQualityScorer
    
    scorer = FaceQualityScorer()
    quality = scorer.calculate(face_img, det_score, landmarks)
"""

import cv2
import numpy as np

class FaceQualityScorer:
    """
    Unified Face Quality Scoring System.
    
    Combines blur, brightness, pose, and detection confidence
    into a single quality score between 0.0 and 1.0.
    """
    
    # Default weights (can be customized)
    WEIGHTS = {
        'blur': 0.30,        # Sharpness is important
        'brightness': 0.20,  # Lighting conditions
        'pose': 0.25,        # Face angle
        'det_score': 0.25    # Detection confidence
    }
    
    # Normalization constants
    BLUR_MAX = 500.0        # Laplacian variance saturation
    BLUR_MIN = 50.0         # Minimum acceptable blur
    
    def __init__(self, weights=None):
        """
        Initialize scorer with optional custom weights.
        
        Args:
            weights: Dict with keys 'blur', 'brightness', 'pose', 'det_score'
        """
        if weights:
            self.weights = weights
        else:
            self.weights = self.WEIGHTS.copy()
    
    def calculate_blur_score(self, face_img):
        """
        Calculate blur score using Laplacian variance.
        
        Args:
            face_img: Aligned face image
            
        Returns:
            Normalized blur score (0.0 = blurry, 1.0 = sharp)
        """
        if face_img is None or face_img.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range
        score = (variance - self.BLUR_MIN) / (self.BLUR_MAX - self.BLUR_MIN)
        return max(0.0, min(1.0, score))
    
    def calculate_brightness_score(self, face_img):
        """
        Calculate brightness score.
        
        Args:
            face_img: Aligned face image
            
        Returns:
            Brightness score (0.0 = too dark/bright, 1.0 = ideal)
        """
        if face_img is None or face_img.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        mean_brightness = np.mean(gray)
        
        # Ideal brightness range: 80-180
        ideal_low = 80
        ideal_high = 180
        ideal_center = (ideal_low + ideal_high) / 2
        
        if ideal_low <= mean_brightness <= ideal_high:
            # Within ideal range - calculate how close to center
            distance = abs(mean_brightness - ideal_center)
            max_distance = (ideal_high - ideal_low) / 2
            score = 1.0 - (distance / max_distance) * 0.3  # Max penalty 30%
        elif mean_brightness < ideal_low:
            # Too dark
            score = mean_brightness / ideal_low * 0.7
        else:
            # Too bright
            score = max(0, 1.0 - (mean_brightness - ideal_high) / (255 - ideal_high)) * 0.7
        
        return max(0.0, min(1.0, score))
    
    def calculate_pose_score(self, landmarks):
        """
        Calculate pose score based on landmark positions.
        
        Args:
            landmarks: 5x2 array of facial landmarks
            
        Returns:
            Pose score (0.0 = extreme pose, 1.0 = frontal)
        """
        if landmarks is None or len(landmarks) < 5:
            return 0.5  # Unknown pose
        
        # Get eye and nose positions
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        
        # Eye distance
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist < 1e-6:
            return 0.0
        
        # Check horizontal symmetry (nose should be between eyes)
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_offset = abs(nose[0] - eye_center_x) / eye_dist
        
        # Ideal: nose_offset close to 0
        horizontal_score = max(0, 1.0 - nose_offset * 2)
        
        # Check vertical position (eye-nose distance ratio)
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        eye_nose_dist = nose[1] - eye_center_y
        vertical_ratio = eye_nose_dist / eye_dist
        
        # Ideal ratio: around 0.5-0.8
        if 0.4 <= vertical_ratio <= 1.0:
            vertical_score = 1.0
        else:
            vertical_score = max(0, 1.0 - abs(vertical_ratio - 0.7))
        
        return (horizontal_score * 0.6 + vertical_score * 0.4)
    
    def calculate(self, face_img=None, det_score=1.0, landmarks=None, 
                  blur_score=None, brightness_score=None, pose_score=None):
        """
        Calculate unified quality score.
        
        Args:
            face_img: Aligned face image (optional if individual scores provided)
            det_score: Detection confidence (0.0 - 1.0)
            landmarks: 5x2 facial landmarks (optional)
            blur_score: Pre-calculated blur score (optional)
            brightness_score: Pre-calculated brightness score (optional)
            pose_score: Pre-calculated pose score (optional)
            
        Returns:
            dict with individual scores and unified 'quality_score'
        """
        # Calculate individual scores
        if blur_score is None and face_img is not None:
            blur_score = self.calculate_blur_score(face_img)
        elif blur_score is None:
            blur_score = 0.5
        
        if brightness_score is None and face_img is not None:
            brightness_score = self.calculate_brightness_score(face_img)
        elif brightness_score is None:
            brightness_score = 0.5
        
        if pose_score is None and landmarks is not None:
            pose_score = self.calculate_pose_score(landmarks)
        elif pose_score is None:
            pose_score = 0.5
        
        # Ensure det_score is in valid range
        det_score = max(0.0, min(1.0, det_score))
        
        # Calculate weighted average
        quality_score = (
            self.weights['blur'] * blur_score +
            self.weights['brightness'] * brightness_score +
            self.weights['pose'] * pose_score +
            self.weights['det_score'] * det_score
        )
        
        return {
            'quality_score': quality_score,
            'blur_score': blur_score,
            'brightness_score': brightness_score,
            'pose_score': pose_score,
            'det_score': det_score
        }

# Convenience function
def calculate_face_quality(face_img, det_score=1.0, landmarks=None):
    """
    Quick quality calculation.
    
    Returns:
        float: Quality score between 0.0 and 1.0
    """
    scorer = FaceQualityScorer()
    result = scorer.calculate(face_img, det_score, landmarks)
    return result['quality_score']
