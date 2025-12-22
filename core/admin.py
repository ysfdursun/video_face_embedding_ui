from django.contrib import admin
from .models import Actor, Movie, MovieCast,FaceRecognitionSettings,FaceGroup



admin.site.register(Actor)
admin.site.register(Movie)
admin.site.register(MovieCast)
admin.site.register(FaceRecognitionSettings)
admin.site.register(FaceGroup)
# Register your models here.
