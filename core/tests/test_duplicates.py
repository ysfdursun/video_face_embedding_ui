# -*- coding: utf-8 -*-
"""
Unit tests for DuplicateDetector module.

Tests pHash computation and embedding similarity functions.
"""

import unittest
import numpy as np
import cv2
from core.quality.duplicate_detector import DuplicateDetector, get_duplicate_detector


class TestPHashComputation(unittest.TestCase):
    """Tests for perceptual hash computation."""
    
    def setUp(self):
        self.detector = get_duplicate_detector()
    
    def test_phash_identical_images(self):
        """Identical images should have same hash."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        hash1 = self.detector.compute_phash(img)
        hash2 = self.detector.compute_phash(img.copy())
        
        self.assertEqual(hash1, hash2, "Identical images should have same hash")
    
    def test_phash_similar_images(self):
        """Slightly modified images should have similar hashes."""
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        
        # Add small noise
        noisy = img.copy()
        noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
        noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        hash1 = self.detector.compute_phash(img)
        hash2 = self.detector.compute_phash(noisy)
        
        distance = self.detector.hamming_distance(hash1, hash2)
        self.assertLess(distance, 15, "Similar images should have low Hamming distance")
    
    def test_phash_different_images(self):
        """Very different images should have different hashes."""
        # Create structurally different images
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img1[0:50, 0:50] = 255  # Top-left white
        
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2[50:100, 50:100] = 255  # Bottom-right white
        
        hash1 = self.detector.compute_phash(img1)
        hash2 = self.detector.compute_phash(img2)
        
        distance = self.detector.hamming_distance(hash1, hash2)
        # Just verify hashes are different, not necessarily by large margin
        self.assertGreater(distance, 0, "Structurally different images should have different hashes")
    
    def test_phash_empty_image(self):
        """Empty image should return empty hash."""
        empty = np.array([])
        hash_val = self.detector.compute_phash(empty)
        self.assertEqual(hash_val, "")
    
    def test_phash_grayscale(self):
        """Grayscale image should be handled correctly."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        hash_val = self.detector.compute_phash(gray)
        self.assertEqual(len(hash_val), 16, "Hash should be 16 hex chars (64 bits)")


class TestHammingDistance(unittest.TestCase):
    """Tests for Hamming distance calculation."""
    
    def setUp(self):
        self.detector = get_duplicate_detector()
    
    def test_hamming_identical(self):
        """Identical hashes should have distance 0."""
        distance = self.detector.hamming_distance("0000000000000000", "0000000000000000")
        self.assertEqual(distance, 0)
    
    def test_hamming_one_bit(self):
        """Single bit difference should have distance 1."""
        distance = self.detector.hamming_distance("0000000000000000", "0000000000000001")
        self.assertEqual(distance, 1)
    
    def test_hamming_invalid_hash(self):
        """Invalid hash should return max distance."""
        distance = self.detector.hamming_distance("", "0000000000000000")
        self.assertEqual(distance, 64)


class TestNearDuplicate(unittest.TestCase):
    """Tests for near-duplicate detection."""
    
    def setUp(self):
        self.detector = get_duplicate_detector()
    
    def test_near_duplicate_true(self):
        """Very similar hashes should be detected as near-duplicates."""
        # Hamming distance < 5
        result = self.detector.is_near_duplicate(
            "0000000000000000",
            "0000000000000003"  # Only 2 bits different
        )
        self.assertTrue(result)
    
    def test_near_duplicate_false(self):
        """Different hashes should not be near-duplicates."""
        result = self.detector.is_near_duplicate(
            "0000000000000000",
            "00000000000000ff"  # 8 bits different
        )
        self.assertFalse(result)


class TestEmbeddingSimilarity(unittest.TestCase):
    """Tests for embedding cosine similarity."""
    
    def setUp(self):
        self.detector = get_duplicate_detector()
    
    def test_cosine_identical(self):
        """Identical normalized embeddings should have similarity 1.0."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        sim = self.detector.cosine_similarity(emb, emb)
        self.assertAlmostEqual(sim, 1.0, places=5)
    
    def test_cosine_orthogonal(self):
        """Orthogonal embeddings should have similarity ~0."""
        emb1 = np.zeros(512, dtype=np.float32)
        emb1[0] = 1.0
        
        emb2 = np.zeros(512, dtype=np.float32)
        emb2[1] = 1.0
        
        sim = self.detector.cosine_similarity(emb1, emb2)
        self.assertAlmostEqual(sim, 0.0, places=5)
    
    def test_cosine_opposite(self):
        """Opposite embeddings should have similarity -1.0."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        sim = self.detector.cosine_similarity(emb, -emb)
        self.assertAlmostEqual(sim, -1.0, places=5)
    
    def test_cosine_none(self):
        """None embeddings should return 0."""
        sim = self.detector.cosine_similarity(None, np.ones(512))
        self.assertEqual(sim, 0.0)


class TestRedundancyDetection(unittest.TestCase):
    """Tests for embedding redundancy detection."""
    
    def setUp(self):
        self.detector = get_duplicate_detector()
    
    def test_redundant_same_embedding(self):
        """Same embedding in list should be detected as redundant."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        existing = [emb.copy()]
        is_redundant, sim = self.detector.is_redundant(emb, existing)
        
        self.assertTrue(is_redundant)
        self.assertGreater(sim, 0.95)
    
    def test_not_redundant_different(self):
        """Different embeddings should not be redundant."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        is_redundant, sim = self.detector.is_redundant(emb2, [emb1])
        
        # Random embeddings typically have low similarity
        self.assertLess(sim, 0.5)
    
    def test_empty_list(self):
        """Empty existing list should not be redundant."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        is_redundant, sim = self.detector.is_redundant(emb, [])
        
        self.assertFalse(is_redundant)
        self.assertEqual(sim, 0.0)


class TestClassification(unittest.TestCase):
    """Tests for embedding similarity classification."""
    
    def setUp(self):
        self.detector = get_duplicate_detector()
    
    def test_classify_redundant(self):
        """High similarity should be classified as redundant."""
        result = self.detector.classify_embedding_similarity(0.98)
        self.assertEqual(result, 'redundant')
    
    def test_classify_same_person(self):
        """Medium similarity should be classified as same_person."""
        result = self.detector.classify_embedding_similarity(0.85)
        self.assertEqual(result, 'same_person')
    
    def test_classify_different(self):
        """Low similarity should be classified as different_person."""
        result = self.detector.classify_embedding_similarity(0.5)
        self.assertEqual(result, 'different_person')


if __name__ == '__main__':
    unittest.main()
