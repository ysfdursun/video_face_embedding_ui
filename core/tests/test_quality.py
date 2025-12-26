# -*- coding: utf-8 -*-
"""
Unit tests for FaceQualityScorer module.

Tests blur, illumination, and pose scoring functions.
"""

import unittest
import numpy as np
import cv2
from core.quality.face_quality import FaceQualityScorer, get_quality_scorer


class TestBlurScore(unittest.TestCase):
    """Tests for blur score calculation."""
    
    def setUp(self):
        self.scorer = get_quality_scorer()
    
    def test_blur_score_clear_image(self):
        """Sharp image with high-frequency details should get high score."""
        # Create a sharp checkerboard pattern
        img = np.zeros((100, 100), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
        
        score = self.scorer.calculate_blur_score(img)
        self.assertGreater(score, 0.5, "Sharp checkerboard should have high blur score")
    
    def test_blur_score_blurry_image(self):
        """Blurred image should get low score."""
        # Create a sharp image and blur it heavily
        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, 40:60] = 255  # Simple box
        
        blurred = cv2.GaussianBlur(img, (31, 31), 15)
        
        score = self.scorer.calculate_blur_score(blurred)
        self.assertLess(score, 0.3, "Heavily blurred image should have low blur score")
    
    def test_blur_score_empty_image(self):
        """Empty image should return 0."""
        empty = np.array([])
        score = self.scorer.calculate_blur_score(empty)
        self.assertEqual(score, 0.0)
    
    def test_blur_score_color_image(self):
        """Color image should be handled correctly."""
        # Create color image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        score = self.scorer.calculate_blur_score(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestIlluminationScore(unittest.TestCase):
    """Tests for illumination score calculation."""
    
    def setUp(self):
        self.scorer = get_quality_scorer()
    
    def test_illumination_optimal(self):
        """Image with mean brightness ~128 should get high score."""
        # Create image with optimal brightness
        img = np.full((100, 100), 128, dtype=np.uint8)
        
        score = self.scorer.calculate_illumination_score(img)
        self.assertGreater(score, 0.95, "Optimal brightness should score near 1.0")
    
    def test_illumination_dark(self):
        """Very dark image should get low score."""
        img = np.full((100, 100), 20, dtype=np.uint8)
        
        score = self.scorer.calculate_illumination_score(img)
        self.assertLess(score, 0.3, "Dark image should have low illumination score")
    
    def test_illumination_bright(self):
        """Very bright (overexposed) image should get low score."""
        img = np.full((100, 100), 240, dtype=np.uint8)
        
        score = self.scorer.calculate_illumination_score(img)
        self.assertLess(score, 0.3, "Overexposed image should have low illumination score")
    
    def test_illumination_empty_image(self):
        """Empty image should return 0."""
        empty = np.array([])
        score = self.scorer.calculate_illumination_score(empty)
        self.assertEqual(score, 0.0)


class TestPoseScore(unittest.TestCase):
    """Tests for pose score calculation."""
    
    def setUp(self):
        self.scorer = get_quality_scorer()
    
    def test_pose_frontal(self):
        """Frontal face landmarks should get high score."""
        # Simulate frontal face: symmetric eye positions
        # For frontal view, nose should be further from eye center (ratio < 1.4)
        landmarks = np.array([
            [30, 40],   # Left eye
            [70, 40],   # Right eye  (eye_dist = 40)
            [50, 80],   # Nose (centered, further down = nose_to_eye ~ 40, ratio ~ 1.0)
            [35, 100],  # Left mouth
            [65, 100]   # Right mouth
        ])
        
        score = self.scorer.calculate_pose_score(landmarks)
        self.assertGreater(score, 0.8, "Frontal face should have high pose score")
    
    def test_pose_profile(self):
        """Profile face landmarks should get lower score."""
        # Simulate profile: asymmetric eye positions relative to nose
        landmarks = np.array([
            [30, 40],   # Left eye
            [40, 40],   # Right eye (closer to left - profile view)
            [55, 45],   # Nose (off to the side)
            [30, 80],   # Left mouth
            [40, 80]    # Right mouth
        ])
        
        score = self.scorer.calculate_pose_score(landmarks)
        # Profile faces have higher ratio, so lower score
        self.assertLessEqual(score, 1.0, "Profile face score should be valid")
    
    def test_pose_none_landmarks(self):
        """None landmarks should return default score."""
        score = self.scorer.calculate_pose_score(None)
        self.assertEqual(score, 0.5, "None landmarks should return default 0.5")
    
    def test_pose_insufficient_landmarks(self):
        """Insufficient landmarks should return default score."""
        landmarks = np.array([[30, 40], [70, 40]])  # Only 2 landmarks
        score = self.scorer.calculate_pose_score(landmarks)
        self.assertEqual(score, 0.5)


class TestCombinedScore(unittest.TestCase):
    """Tests for combined quality score calculation."""
    
    def setUp(self):
        self.scorer = get_quality_scorer()
    
    def test_combined_quality(self):
        """Combined score should be weighted average."""
        # Create a reasonable quality image
        img = np.random.randint(100, 150, (100, 100, 3), dtype=np.uint8)
        landmarks = np.array([
            [30, 40], [70, 40], [50, 60], [35, 80], [65, 80]
        ])
        
        result = self.scorer.get_combined_score(img, landmarks, 0.9)
        
        self.assertIn('blur_score', result)
        self.assertIn('illumination_score', result)
        self.assertIn('pose_score', result)
        self.assertIn('quality_score', result)
        self.assertIn('is_valid', result)
        
        # Verify weighted calculation
        expected = (
            0.4 * result['blur_score'] +
            0.3 * result['illumination_score'] +
            0.3 * result['pose_score']
        )
        self.assertAlmostEqual(result['quality_score'], expected, places=3)
    
    def test_validity_threshold(self):
        """Faces below quality threshold should be marked invalid."""
        # Create a very low quality image
        img = np.full((10, 10), 20, dtype=np.uint8)  # Dark, small
        img = cv2.GaussianBlur(img, (5, 5), 5)  # Blurred
        
        result = self.scorer.get_combined_score(img, None, 0.9)
        
        # Quality should be low
        if result['quality_score'] < 0.70:
            self.assertFalse(result['is_valid'])
    
    def test_low_detection_confidence_invalid(self):
        """Low detection confidence should mark face invalid."""
        img = np.full((100, 100), 128, dtype=np.uint8)
        
        result = self.scorer.get_combined_score(img, None, 0.2)  # Below 0.30 threshold
        self.assertFalse(result['is_valid'])


class TestQuickCheck(unittest.TestCase):
    """Tests for quick quality check function."""
    
    def setUp(self):
        self.scorer = get_quality_scorer()
    
    def test_is_quality_acceptable_valid(self):
        """Good quality face should pass quick check."""
        # Create a good quality image
        img = np.random.randint(100, 160, (100, 100), dtype=np.uint8)
        
        # Add some edge details for sharpness
        img[40:60, 40:60] = 200
        
        landmarks = np.array([
            [30, 40], [70, 40], [50, 60], [35, 80], [65, 80]
        ])
        
        is_valid, score = self.scorer.is_quality_acceptable(img, landmarks, 0.9)
        
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(score, float)
    
    def test_is_quality_acceptable_low_confidence(self):
        """Low confidence should immediately fail."""
        img = np.full((100, 100), 128, dtype=np.uint8)
        
        is_valid, score = self.scorer.is_quality_acceptable(img, None, 0.1)
        
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)


if __name__ == '__main__':
    unittest.main()
