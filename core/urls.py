# -*- coding: utf-8 -*-
from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.welcome, name='welcome'),
    path('home/', views.home, name='home'),
    path('label/all/', views.label_all_faces, name='label_all_faces'),
    path('label/list/', views.list_unlabeled_faces, name='list_unlabeled_faces'),
    path('label/delete_single/', views.delete_single_face, name='delete_single_face'),
    path('label/delete_movie/', views.delete_movie, name='delete_movie'),
    path('process/<str:movie_filename>/', views.processing_page, name='processing_page'),
    path('stream/<str:movie_filename>/', views.stream_video_processing, name='stream_video_processing'),
    path('image/optimize/', views.serve_optimized_image, name='serve_optimized_image'),
    path('actors/', views.actors_dashboard, name='actors_dashboard'),
    path('actors/<str:actor_name>/', views.actor_detail, name='actor_detail'),
    path('api/upload-photo/', views.upload_photo, name='upload_photo'),
    path('movies/', views.movies_dashboard, name='movies_dashboard'),
    path('movies/<int:movie_id>/', views.movie_detail, name='movie_detail'),
    path('api/movie/create/', views.MovieCreateAPI.as_view(), name='api_create_movie'),
    path('api/actor/create/', views.ActorCreateAPI.as_view(), name='api_create_actor'),
    path('api/actor/delete/', views.ActorDeleteAPI.as_view(), name='api_delete_actor'),
    path('api/cast/manage/', views.CastManageAPI.as_view(), name='api_cast_manage'),
    path('api/actors/search/', views.ActorSearchAPI.as_view(), name='api_actor_search'),
    path('api/processing-status/', views.ProcessingStatusAPI.as_view(), name='api_processing_status'),
]
