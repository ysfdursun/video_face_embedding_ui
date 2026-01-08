# -*- coding: utf-8 -*-
from django.urls import path
from . import views
from core.views import analysis_views

app_name = 'core'

urlpatterns = [
    path('', views.welcome, name='welcome'),
    path('home/', views.home, name='home'),
    
    # Labeller / Photo Tool
    path('labeller/', views.labeller_page, name='labeller_page'),
    path('api/labeller/analyze/', views.api_analyze_photo, name='api_analyze_photo'),
    path('api/labeller/enroll/', views.api_enroll_face, name='api_enroll_face'),
    path('settings/', views.settings_view, name='settings'),
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
    path('movies/<int:movie_id>/delete/', views.delete_movie_view, name='delete_movie_view'),
    path('api/movie/create/', views.MovieCreateAPI.as_view(), name='api_create_movie'),
    path('api/actor/create/', views.ActorCreateAPI.as_view(), name='api_create_actor'),
    path('api/actor/delete/', views.ActorDeleteAPI.as_view(), name='api_delete_actor'),
    path('api/cast/manage/', views.CastManageAPI.as_view(), name='api_cast_manage'),
    path('api/actors/search/', views.ActorSearchAPI.as_view(), name='api_actor_search'),
    path('api/movies/search/', views.api_search_movies, name='api_search_movies'),
    path('api/processing-status/', views.ProcessingStatusAPI.as_view(), name='api_processing_status'),
    path('api/profile/update/', views.update_profile_photo, name='update_profile_photo'),
    path('api/profile/remove/', views.remove_profile_photo, name='remove_profile_photo'),
    path('api/embedding/toggle/', views.toggle_embedding, name='toggle_embedding'),
    # Recognition
    path('recognition/', views.recognition_page, name='recognition_page'),
    path('recognition/upload/', views.recognition_upload, name='recognition_upload'),
    path('recognition/stream/<str:session_id>/', views.recognition_stream, name='recognition_stream'),
    path('recognition/status/', views.api_recognition_status, name='api_recognition_status'),
    path('recognition/stream/movie/<str:movie_filename>/', views.stream_video_recognition, name='stream_video_recognition_existing'),
    path('recognition/enroll/', views.recognition_enroll_guest, name='recognition_enroll_guest'),
    # Lookalike
    path('lookalike/', views.lookalike_page, name='lookalike_page'),
    path('api/lookalike/analyze/', views.api_analyze_lookalike, name='api_analyze_lookalike'),
    # Analysis
    path('analysis/', analysis_views.analysis_dashboard, name='analysis_dashboard'),
    path('api/analysis/run/', analysis_views.api_run_analysis, name='api_run_analysis'),
]
