# -*- coding: utf-8 -*-
from django.db import models

class Actor(models.Model):
    name = models.CharField(max_length=255, unique=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.name


class Movie(models.Model):
    title = models.CharField(max_length=500, unique=True)
    
    class Meta:
        ordering = ['title']
    
    def __str__(self):
        return self.title


class MovieCast(models.Model):
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='cast_members')
    actor = models.ForeignKey(Actor, on_delete=models.CASCADE)
    
    class Meta:
        unique_together = ('movie', 'actor')
        ordering = ['movie', 'actor']
    
    
    def __str__(self):
        return f"{self.movie.title} - {self.actor.name}"


class FaceRecognitionSettings(models.Model):
    """
    Global settings for face recognition pipeline.
    Singleton pattern implementation.
    """
    # Detection settings
    detection_threshold = models.FloatField(default=0.5, help_text="Minimum confidence for detecting a face (0.0 - 1.0)")
    grouping_threshold = models.FloatField(default=0.45, help_text="Minimum similarity to group faces together (0.0 - 1.0)")
    min_face_size = models.IntegerField(default=80, help_text="Minimum face height/width in pixels")
    
    # Performance settings
    frame_skip_extract = models.IntegerField(default=10, help_text="Process every Nth frame for extraction")
    frame_skip_group = models.IntegerField(default=5, help_text="Process every Nth extracted face for grouping")
    gpu_enabled = models.BooleanField(default=True, help_text="Attempt to use GPU if available")
    
    # Quality & Classification settings
    quality_threshold = models.FloatField(default=0.50, help_text="Minimum combined quality score to keep a face")
    redundancy_threshold = models.FloatField(default=0.85, help_text="Embedding similarity above this = too similar, skip")

    def save(self, *args, **kwargs):
        if not self.pk and FaceRecognitionSettings.objects.exists():
            # If you're trying to save a new instance, but one exists, update the existing one
            return
        return super(FaceRecognitionSettings, self).save(*args, **kwargs)

    @classmethod
    def get_settings(cls):
        obj, created = cls.objects.get_or_create(pk=1)
        return obj

    def __str__(self):
        return "Face Recognition Global Settings"


class FaceGroup(models.Model):
    """
    Metadata for a face group on the filesystem.
    This effectively extends the 'folder' concept with rich data.
    """
    RISK_LEVEL_CHOICES = [
        ('LOW', 'Low Risk (High Confidence)'),
        ('MEDIUM', 'Medium Risk'),
        ('HIGH', 'High Risk (Low Confidence)'),
    ]

    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='face_groups')
    group_id = models.CharField(max_length=50, help_text="Folder name e.g. celebrity_001")
    name = models.CharField(max_length=255, blank=True, help_text="Optional assigned name (actor name)")
    is_labeled = models.BooleanField(default=False, help_text="Whether this group has been labeled with an identity")
    
    # Stats
    total_faces = models.IntegerField(default=0, help_text="Total number of faces in this group")
    face_count = models.IntegerField(default=0)  # Kept for backward compatibility
    avg_confidence = models.FloatField(default=0.0, help_text="Average similarity score of faces in this group")
    avg_quality = models.FloatField(default=0.0, help_text="Average quality score of faces in this group")
    
    # Multi-template representation (serialized numpy array)
    representative_embedding = models.BinaryField(null=True, blank=True, help_text="Serialized representative embedding(s)")
    
    risk_level = models.CharField(max_length=10, choices=RISK_LEVEL_CHOICES, default='HIGH')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('movie', 'group_id')
        ordering = ['movie', 'group_id']

    def update_risk_level(self):
        """Auto-calculate risk based on confidence."""
        if self.avg_confidence > 0.70:
            self.risk_level = 'LOW'
        elif self.avg_confidence > 0.55:
            self.risk_level = 'MEDIUM'
        else:
            self.risk_level = 'HIGH'
        self.save()
    
    def set_representative_embedding(self, embedding):
        """Serialize and store numpy embedding."""
        import numpy as np
        if embedding is not None:
            self.representative_embedding = embedding.astype(np.float32).tobytes()
    
    def get_representative_embedding(self):
        """Deserialize stored embedding."""
        import numpy as np
        if self.representative_embedding:
            return np.frombuffer(self.representative_embedding, dtype=np.float32)
        return None

    def __str__(self):
        return f"{self.movie.title} - {self.group_id} ({self.risk_level})"


class FaceDetection(models.Model):
    """
    Individual face detection with quality metrics.
    
    Stores per-face data including:
    - Quality scores (blur, illumination, pose)
    - 512-D embedding for recognition
    - Perceptual hash for duplicate detection
    - Validity and outlier flags
    """
    
    # Source information
    source_image = models.CharField(max_length=500, help_text="Path to source image or video frame")
    frame_number = models.IntegerField(null=True, blank=True, help_text="Frame number if from video")
    bbox = models.JSONField(default=list, help_text="Bounding box [x1, y1, x2, y2]")
    
    # 512-D normalized embedding (stored as binary for efficiency)
    embedding = models.BinaryField(null=True, blank=True, help_text="512-D ArcFace embedding")
    
    # Quality metrics
    quality_score = models.FloatField(default=0.0, help_text="Combined quality score (0.0-1.0)")
    blur_score = models.FloatField(default=0.0, help_text="Blur score from Laplacian variance")
    illumination_score = models.FloatField(default=0.0, help_text="Illumination score from brightness analysis")
    pose_score = models.FloatField(default=0.0, help_text="Pose score from landmark symmetry")
    detection_confidence = models.FloatField(default=0.0, help_text="Detection model confidence")
    
    # Duplicate detection
    image_hash = models.CharField(max_length=64, blank=True, help_text="Perceptual hash (pHash)")
    
    # Flags
    is_valid = models.BooleanField(default=True, help_text="Passes quality thresholds")
    is_outlier = models.BooleanField(default=False, help_text="Marked as outlier from group")
    
    # Pose estimation (optional, from PnP)
    yaw = models.FloatField(null=True, blank=True, help_text="Horizontal rotation in degrees")
    pitch = models.FloatField(null=True, blank=True, help_text="Vertical rotation in degrees")
    roll = models.FloatField(null=True, blank=True, help_text="In-plane rotation in degrees")
    
    # Relationship to group
    face_group = models.ForeignKey(
        FaceGroup, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='detections'
    )
    
    # Cropped face image path (optional)
    cropped_image_path = models.CharField(max_length=500, blank=True, help_text="Path to saved cropped face")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['image_hash']),
            models.Index(fields=['is_valid', 'is_outlier']),
            models.Index(fields=['face_group']),
        ]

    def set_embedding(self, embedding):
        """Serialize and store numpy embedding."""
        import numpy as np
        if embedding is not None:
            self.embedding = embedding.astype(np.float32).tobytes()
    
    def get_embedding(self):
        """Deserialize stored embedding."""
        import numpy as np
        if self.embedding:
            return np.frombuffer(self.embedding, dtype=np.float32)
        return None

    def __str__(self):
        return f"Face {self.id} - Q:{self.quality_score:.2f} - {'Valid' if self.is_valid else 'Invalid'}"

