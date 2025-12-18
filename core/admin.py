from django.contrib import admin
from .models import Actor, Movie, MovieCast



admin.site.register(Actor)
admin.site.register(Movie)
admin.site.register(MovieCast)

# Register your models here.
