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
    detection_threshold = models.FloatField(default=0.5, help_text="Minimum confidence for detecting a face (0.0 - 1.0)")
    grouping_threshold = models.FloatField(default=0.45, help_text="Minimum similarity to group faces together (0.0 - 1.0)")
    min_face_size = models.IntegerField(default=80, help_text="Minimum face height/width in pixels")
    frame_skip_extract = models.IntegerField(default=10, help_text="Process every Nth frame for extraction")
    frame_skip_group = models.IntegerField(default=5, help_text="Process every Nth extracted face for grouping")
    gpu_enabled = models.BooleanField(default=True, help_text="Attempt to use GPU if available")

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
    
    # Stats
    face_count = models.IntegerField(default=0)
    avg_confidence = models.FloatField(default=0.0, help_text="Average similarity score of faces in this group")
    
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

    def __str__(self):
        return f"{self.movie.title} - {self.group_id} ({self.risk_level})"
