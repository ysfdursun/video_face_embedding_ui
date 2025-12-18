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