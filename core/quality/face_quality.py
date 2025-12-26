# -*- coding: utf-8 -*-
"""
Face Quality Scoring Module

Implements quality metrics for face detection:
- Blur Score (40%): Laplacian variance
- Illumination Score (30%): Gaussian brightness
- Pose Score (30%): 5-point landmark symmetry
"""

import cv2
import numpy as np
import math


class FaceQualityScorer:
    """
    Calculates quality metrics for detected faces.
    
    Quality thresholds follow production face recognition standards:
    - Minimum combined quality: 0.70
    - Detection confidence minimum: 0.30
    """
    
    # Weights for combined score
    BLUR_WEIGHT = 0.40
    ILLUMINATION_WEIGHT = 0.30
    POSE_WEIGHT = 0.30
    
    # Blur thresholds (Laplacian variance)
    # Lowered for video processing (video frames tend to have more blur)
    BLUR_MIN = 30.0      # Below this = unusable (was 50)
    BLUR_MAX = 250.0     # Above this = high quality (was 300)
    BLUR_RANGE = BLUR_MAX - BLUR_MIN
    
    # Illumination parameters
    ILLUMINATION_OPTIMAL = 128.0
    ILLUMINATION_SIGMA = 40.0
    
    # Pose thresholds
    POSE_FRONTAL_RATIO = 1.4   # Below = frontal (best)
    POSE_TILT_RATIO = 2.0      # Above = profile (low quality)
    
    # Detection confidence minimum
    MIN_DETECTION_CONFIDENCE = 0.30
    
    # Combined quality minimum (lowered from 0.70 for better video coverage)
    MIN_QUALITY_SCORE = 0.50
    
    def __init__(self):
        pass
    
    def calculate_blur_score(self, face_img: np.ndarray) -> float:
        """
        Calculate blur score using Laplacian variance.
        
        Args:
            face_img: Face image (BGR or grayscale)
            
        Returns:
            Normalized score [0.0, 1.0] where 1.0 = sharp
        """
        if face_img is None or face_img.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Normalize to [0, 1]
        if laplacian_var < self.BLUR_MIN:
            return 0.0
        elif laplacian_var > self.BLUR_MAX:
            return 1.0
        else:
            return (laplacian_var - self.BLUR_MIN) / self.BLUR_RANGE
    
    def calculate_illumination_score(self, face_img: np.ndarray) -> float:
        """
        Calculate illumination score using Gaussian-based brightness analysis.
        
        Optimal brightness is around 128 (middle gray). Too dark or too bright
        images receive lower scores.
        
        Args:
            face_img: Face image (BGR or grayscale)
            
        Returns:
            Normalized score [0.0, 1.0] where 1.0 = optimal brightness
        """
        if face_img is None or face_img.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Calculate mean brightness
        mean_brightness = float(np.mean(gray))
        
        # Gaussian scoring: optimal at 128, falls off with sigma=40
        score = math.exp(
            -((mean_brightness - self.ILLUMINATION_OPTIMAL) ** 2) / 
            (2 * self.ILLUMINATION_SIGMA ** 2)
        )
        
        return score
    
    def calculate_pose_score(self, landmarks: np.ndarray) -> float:
        """
        Calculate pose score based on 5-point landmark symmetry.
        
        Landmarks expected: [Left Eye, Right Eye, Nose, Left Mouth, Right Mouth]
        
        Uses the ratio of eye distance to nose-eye_center distance
        to estimate face angle (frontal vs profile).
        
        Args:
            landmarks: 5x2 array of landmark coordinates
            
        Returns:
            Normalized score [0.0, 1.0] where 1.0 = perfectly frontal
        """
        if landmarks is None or len(landmarks) < 5:
            return 0.5  # Default score if landmarks unavailable
        
        try:
            # Extract key points
            left_eye = np.array(landmarks[0])
            right_eye = np.array(landmarks[1])
            nose = np.array(landmarks[2])
            
            # Calculate eye center
            eye_center = (left_eye + right_eye) / 2
            
            # Calculate distances
            eye_distance = np.linalg.norm(right_eye - left_eye)
            nose_to_eye_mid = np.linalg.norm(nose - eye_center)
            
            # Avoid division by zero
            if nose_to_eye_mid < 1e-6:
                return 0.5
            
            # Calculate ratio
            ratio = eye_distance / nose_to_eye_mid
            
            # Score based on ratio thresholds
            if ratio < self.POSE_FRONTAL_RATIO:
                # Frontal face - best score
                return 1.0
            elif ratio < self.POSE_TILT_RATIO:
                # Slight tilt - interpolate score
                t = (ratio - self.POSE_FRONTAL_RATIO) / (self.POSE_TILT_RATIO - self.POSE_FRONTAL_RATIO)
                return 1.0 - (0.5 * t)  # Score from 1.0 to 0.5
            else:
                # Profile view - low score
                return max(0.2, 0.5 - (ratio - self.POSE_TILT_RATIO) * 0.1)
                
        except Exception:
            return 0.5  # Default on error
    
    def calculate_pose_from_angles(self, yaw: float, pitch: float, roll: float) -> float:
        """
        Calculate pose score from PnP-estimated angles.
        
        Args:
            yaw: Horizontal rotation in degrees
            pitch: Vertical rotation in degrees  
            roll: In-plane rotation in degrees
            
        Returns:
            Normalized score [0.0, 1.0]
        """
        # Primary penalty from yaw (profile view)
        yaw_penalty = min(1.0, abs(yaw) / 45.0)
        
        # Secondary penalties from pitch and roll
        pitch_penalty = min(0.3, abs(pitch) / 60.0)
        roll_penalty = min(0.2, abs(roll) / 45.0)
        
        # Combined score
        score = max(0.0, 1.0 - yaw_penalty - pitch_penalty - roll_penalty)
        
        return score
    
    def get_combined_score(
        self,
        face_img: np.ndarray,
        landmarks: np.ndarray = None,
        detection_confidence: float = 1.0
    ) -> dict:
        """
        Calculate all quality metrics and combined score.
        
        Args:
            face_img: Face image (BGR)
            landmarks: Optional 5-point landmarks
            detection_confidence: Detection model confidence
            
        Returns:
            Dictionary containing:
            - blur_score: float
            - illumination_score: float
            - pose_score: float
            - quality_score: float (weighted combination)
            - is_valid: bool (passes minimum thresholds)
        """
        # Calculate individual scores
        blur_score = self.calculate_blur_score(face_img)
        illumination_score = self.calculate_illumination_score(face_img)
        pose_score = self.calculate_pose_score(landmarks) if landmarks is not None else 0.5
        
        # Weighted combination
        quality_score = (
            self.BLUR_WEIGHT * blur_score +
            self.ILLUMINATION_WEIGHT * illumination_score +
            self.POSE_WEIGHT * pose_score
        )
        
        # Validity check
        is_valid = (
            quality_score >= self.MIN_QUALITY_SCORE and
            detection_confidence >= self.MIN_DETECTION_CONFIDENCE
        )
        
        return {
            'blur_score': round(blur_score, 4),
            'illumination_score': round(illumination_score, 4),
            'pose_score': round(pose_score, 4),
            'quality_score': round(quality_score, 4),
            'detection_confidence': round(detection_confidence, 4),
            'is_valid': is_valid
        }
    
    def is_quality_acceptable(
        self,
        face_img: np.ndarray,
        landmarks: np.ndarray = None,
        detection_confidence: float = 1.0
    ) -> tuple:
        """
        Quick check if face passes quality threshold.
        
        Returns:
            Tuple of (is_valid: bool, quality_score: float)
        """
        # Fast rejection based on detection confidence
        if detection_confidence < self.MIN_DETECTION_CONFIDENCE:
            return False, 0.0
        
        result = self.get_combined_score(face_img, landmarks, detection_confidence)
        return result['is_valid'], result['quality_score']


# Singleton instance for reuse
_quality_scorer = None

def get_quality_scorer() -> FaceQualityScorer:
    """Get singleton FaceQualityScorer instance."""
    global _quality_scorer
    if _quality_scorer is None:
        _quality_scorer = FaceQualityScorer()
    return _quality_scorer
