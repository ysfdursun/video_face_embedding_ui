# -*- coding: utf-8 -*-
"""
Multi-Template Manager Module

Implements KMeans-based multi-template representation for face groups.
Handles outlier detection based on group mean similarity.
"""

import numpy as np
from typing import List, Tuple, Optional


class TemplateManager:
    """
    Manages multi-template representation for face groups.
    
    Template count rules:
    - < 3 faces: 1 template
    - 3-7 faces: 2 templates
    - >= 8 faces: 3 templates
    
    Each template is a cluster centroid representing a distinct pose/condition.
    """
    
    # Template count thresholds
    MIN_FACES_TWO_TEMPLATES = 3
    MIN_FACES_THREE_TEMPLATES = 8
    MAX_TEMPLATES = 3
    
    # Outlier detection
    OUTLIER_THRESHOLD = 0.70  # Below this = outlier
    
    # KMeans parameters
    MAX_ITERATIONS = 100
    CONVERGENCE_THRESHOLD = 1e-4
    
    def __init__(self):
        pass
    
    def determine_template_count(self, n_faces: int) -> int:
        """
        Determine optimal number of templates based on face count.
        
        Args:
            n_faces: Number of faces in the group
            
        Returns:
            Number of templates to generate (1, 2, or 3)
        """
        if n_faces < self.MIN_FACES_TWO_TEMPLATES:
            return 1
        elif n_faces < self.MIN_FACES_THREE_TEMPLATES:
            return 2
        else:
            return self.MAX_TEMPLATES
    
    def kmeans_clustering(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        max_iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple KMeans clustering for embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            
        Returns:
            Tuple of (centroids, labels)
            - centroids: shape (n_clusters, embedding_dim)
            - labels: shape (n_samples,) cluster assignments
        """
        if max_iter is None:
            max_iter = self.MAX_ITERATIONS
        
        n_samples, dim = embeddings.shape
        
        # Ensure we don't have more clusters than samples
        n_clusters = min(n_clusters, n_samples)
        
        # Initialize centroids using k-means++ style
        centroids = self._init_centroids_plusplus(embeddings, n_clusters)
        
        labels = np.zeros(n_samples, dtype=np.int32)
        
        for _ in range(max_iter):
            # Assignment step
            new_labels = self._assign_clusters(embeddings, centroids)
            
            # Check convergence
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            
            # Update step
            new_centroids = self._update_centroids(embeddings, labels, n_clusters, dim)
            
            # Check centroid convergence
            if np.max(np.abs(centroids - new_centroids)) < self.CONVERGENCE_THRESHOLD:
                centroids = new_centroids
                break
            centroids = new_centroids
        
        # Normalize centroids
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        centroids = centroids / norms
        
        return centroids, labels
    
    def _init_centroids_plusplus(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Initialize centroids using k-means++ algorithm."""
        n_samples, dim = embeddings.shape
        centroids = np.zeros((n_clusters, dim))
        
        # First centroid: random
        idx = np.random.randint(n_samples)
        centroids[0] = embeddings[idx]
        
        # Remaining centroids: probability proportional to distance^2
        for i in range(1, n_clusters):
            # Calculate distances to nearest centroid
            dists = np.zeros(n_samples)
            for j in range(n_samples):
                min_dist = float('inf')
                for k in range(i):
                    # Using 1 - cosine similarity as distance
                    sim = np.dot(embeddings[j], centroids[k])
                    dist = 1 - sim
                    min_dist = min(min_dist, dist)
                dists[j] = min_dist ** 2
            
            # Normalize to probability
            if dists.sum() > 0:
                probs = dists / dists.sum()
                idx = np.random.choice(n_samples, p=probs)
            else:
                idx = np.random.randint(n_samples)
            
            centroids[i] = embeddings[idx]
        
        return centroids
    
    def _assign_clusters(self, embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each embedding to nearest centroid."""
        # Cosine similarity: embeddings @ centroids.T
        similarities = embeddings @ centroids.T
        return np.argmax(similarities, axis=1)
    
    def _update_centroids(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
        dim: int
    ) -> np.ndarray:
        """Update centroids as mean of assigned embeddings."""
        centroids = np.zeros((n_clusters, dim))
        
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centroids[k] = embeddings[mask].mean(axis=0)
            else:
                # Empty cluster: reinitialize randomly
                centroids[k] = embeddings[np.random.randint(len(embeddings))]
        
        return centroids
    
    def generate_templates(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Generate multi-template representation for a group.
        
        Args:
            embeddings: List of 512-D normalized embeddings
            
        Returns:
            Array of template embeddings, shape (n_templates, 512)
        """
        if not embeddings:
            return np.array([])
        
        emb_array = np.array(embeddings)
        n_faces = len(emb_array)
        n_templates = self.determine_template_count(n_faces)
        
        if n_templates == 1:
            # Single template: mean of all embeddings
            template = emb_array.mean(axis=0)
            template = template / np.linalg.norm(template)
            return template.reshape(1, -1)
        
        # Multiple templates: use KMeans
        centroids, _ = self.kmeans_clustering(emb_array, n_templates)
        return centroids
    
    def calculate_group_mean(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Calculate the mean embedding for a group.
        
        Args:
            embeddings: List of 512-D normalized embeddings
            
        Returns:
            Normalized mean embedding
        """
        if not embeddings:
            return np.zeros(512)
        
        mean = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean
    
    def detect_outliers(
        self,
        embeddings: List[np.ndarray],
        group_mean: np.ndarray = None
    ) -> List[int]:
        """
        Detect outlier embeddings in a group.
        
        An embedding is an outlier if its cosine similarity to the group mean
        is below the threshold (0.70).
        
        Args:
            embeddings: List of 512-D embeddings
            group_mean: Optional pre-computed group mean
            
        Returns:
            List of indices that are outliers
        """
        if len(embeddings) < 2:
            return []
        
        if group_mean is None:
            group_mean = self.calculate_group_mean(embeddings)
        
        outliers = []
        for i, emb in enumerate(embeddings):
            sim = float(np.dot(emb, group_mean))
            if sim < self.OUTLIER_THRESHOLD:
                outliers.append(i)
        
        return outliers
    
    def match_to_templates(
        self,
        embedding: np.ndarray,
        templates: np.ndarray,
        threshold: float = 0.65
    ) -> Tuple[Optional[int], float]:
        """
        Match an embedding against multi-templates.
        
        Args:
            embedding: 512-D normalized embedding
            templates: Array of template embeddings
            threshold: Minimum similarity for match
            
        Returns:
            Tuple of (matched_index or None, best_similarity)
        """
        if templates is None or len(templates) == 0:
            return None, 0.0
        
        similarities = templates @ embedding
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        
        if best_sim >= threshold:
            return best_idx, best_sim
        return None, best_sim


# Singleton instance
_template_manager = None

def get_template_manager() -> TemplateManager:
    """Get singleton TemplateManager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager
