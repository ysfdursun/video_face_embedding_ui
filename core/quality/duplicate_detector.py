# -*- coding: utf-8 -*-
"""
Duplicate Detection Module

Implements two-stage duplicate detection:
1. pHash: Perceptual hashing for exact/near duplicates
2. Embedding Similarity: Cosine similarity for redundancy check
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple


class DuplicateDetector:
    """
    Detects duplicate and redundant face images.
    
    Thresholds:
    - pHash Hamming distance < 5: Exact/near duplicate (discard)
    - Embedding cosine > 0.95: Redundant (same frame, discard)
    - Embedding cosine 0.70-0.95: Same person, different image (keep)
    - Embedding cosine < 0.70: Different person or outlier
    """
    
    # pHash parameters
    PHASH_SIZE = 8          # 8x8 hash = 64 bits
    PHASH_HIGHFREQ = 4      # Keep top 4x4 DCT coefficients
    PHASH_THRESHOLD = 5     # Hamming distance for near-duplicate
    
    # Embedding similarity thresholds
    # Lowered from 0.95 to 0.85 for better diversity (avoid overfitting)
    REDUNDANT_THRESHOLD = 0.85    # Same pose/lighting → SKIP
    SAME_PERSON_MIN = 0.70        # Minimum for same person
    SAME_PERSON_MAX = 0.85        # Maximum (above is redundant)
    
    def __init__(self):
        pass
    
    def compute_phash(self, image: np.ndarray) -> str:
        """
        Compute perceptual hash of an image.
        
        Uses DCT-based pHash algorithm:
        1. Resize to 32x32
        2. Compute DCT
        3. Keep low-frequency components
        4. Compute median and binarize
        
        Args:
            image: BGR or grayscale image
            
        Returns:
            64-bit hex string hash
        """
        if image is None or image.size == 0:
            return ""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize for hash computation (32x32 for good DCT)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        
        # Convert to float for DCT
        resized = np.float32(resized)
        
        # Compute 2D DCT
        dct = cv2.dct(resized)
        
        # Keep top-left 8x8 (low frequencies)
        dct_low = dct[:self.PHASH_SIZE, :self.PHASH_SIZE]
        
        # Compute median (excluding DC component)
        dct_flat = dct_low.flatten()
        median = np.median(dct_flat[1:])  # Exclude [0,0] DC component
        
        # Binarize: 1 if above median, 0 otherwise
        bits = (dct_low > median).flatten()
        
        # Convert to hex string
        hash_value = 0
        for bit in bits:
            hash_value = (hash_value << 1) | int(bit)
        
        return format(hash_value, '016x')  # 64 bits = 16 hex chars
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Calculate Hamming distance between two hashes.
        
        Args:
            hash1: First hex hash string
            hash2: Second hex hash string
            
        Returns:
            Number of differing bits
        """
        if not hash1 or not hash2:
            return 64  # Maximum distance if hash is invalid
        
        try:
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            xor = val1 ^ val2
            return bin(xor).count('1')
        except ValueError:
            return 64
    
    def is_near_duplicate(self, hash1: str, hash2: str) -> bool:
        """
        Check if two images are near-duplicates based on pHash.
        
        Returns:
            True if Hamming distance < threshold
        """
        return self.hamming_distance(hash1, hash2) < self.PHASH_THRESHOLD
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Assumes embeddings are already L2-normalized.
        
        Args:
            emb1: First embedding (512-D)
            emb2: Second embedding (512-D)
            
        Returns:
            Similarity score [-1.0, 1.0]
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        # For normalized vectors, cosine similarity = dot product
        return float(np.dot(emb1, emb2))
    
    def is_redundant(self, new_embedding: np.ndarray, existing_embeddings: List[np.ndarray]) -> Tuple[bool, float]:
        """
        Check if a new embedding is redundant (nearly identical to existing ones).
        
        Args:
            new_embedding: The embedding to check
            existing_embeddings: List of existing embeddings in the group
            
        Returns:
            Tuple of (is_redundant: bool, max_similarity: float)
        """
        if not existing_embeddings:
            return False, 0.0
        
        max_sim = 0.0
        for emb in existing_embeddings:
            sim = self.cosine_similarity(new_embedding, emb)
            max_sim = max(max_sim, sim)
            
            # Early exit if clearly redundant
            if sim > self.REDUNDANT_THRESHOLD:
                return True, sim
        
        return False, max_sim
    
    def classify_embedding_similarity(self, similarity: float) -> str:
        """
        Classify the relationship based on embedding similarity.
        
        Returns:
            One of: 'redundant', 'same_person', 'different_person'
        """
        if similarity > self.REDUNDANT_THRESHOLD:
            return 'redundant'
        elif similarity >= self.SAME_PERSON_MIN:
            return 'same_person'
        else:
            return 'different_person'
    
    def find_duplicates_in_batch(
        self,
        images: List[np.ndarray],
        embeddings: List[np.ndarray] = None
    ) -> List[int]:
        """
        Find indices of duplicate images in a batch.
        
        Uses pHash for initial filtering, then embedding similarity.
        
        Args:
            images: List of face images
            embeddings: Optional list of corresponding embeddings
            
        Returns:
            List of indices that are duplicates (should be removed)
        """
        if len(images) < 2:
            return []
        
        # Compute hashes
        hashes = [self.compute_phash(img) for img in images]
        
        duplicate_indices = set()
        n = len(images)
        
        # Find pHash duplicates
        for i in range(n):
            if i in duplicate_indices:
                continue
            for j in range(i + 1, n):
                if j in duplicate_indices:
                    continue
                if self.is_near_duplicate(hashes[i], hashes[j]):
                    duplicate_indices.add(j)  # Keep earlier, remove later
        
        # If embeddings provided, check for embedding redundancy
        if embeddings:
            for i in range(n):
                if i in duplicate_indices:
                    continue
                for j in range(i + 1, n):
                    if j in duplicate_indices:
                        continue
                    sim = self.cosine_similarity(embeddings[i], embeddings[j])
                    if sim > self.REDUNDANT_THRESHOLD:
                        duplicate_indices.add(j)
        
        return sorted(list(duplicate_indices))


# Singleton instance
_duplicate_detector = None

def get_duplicate_detector() -> DuplicateDetector:
    """Get singleton DuplicateDetector instance."""
    global _duplicate_detector
    if _duplicate_detector is None:
        _duplicate_detector = DuplicateDetector()
    return _duplicate_detector
