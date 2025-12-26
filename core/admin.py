from django.contrib import admin
from .models import Actor, Movie, MovieCast, FaceRecognitionSettings, FaceGroup, FaceDetection



admin.site.register(Actor)
admin.site.register(Movie)
admin.site.register(MovieCast)
admin.site.register(FaceRecognitionSettings)
admin.site.register(FaceGroup)
admin.site.register(FaceDetection)
