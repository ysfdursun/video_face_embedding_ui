from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.welcome, name='welcome'),
    path('home/', views.home, name='home'),
    path('label/all/', views.label_all_faces, name='label_all_faces'),
    path('label/list/', views.list_unlabeled_faces, name='list_unlabeled_faces'),
    path('label/delete_single/', views.delete_single_face, name='delete_single_face'),
    path('process/<str:movie_filename>/', views.processing_page, name='processing_page'),
    path('stream/<str:movie_filename>/', views.stream_video_processing, name='stream_video_processing'),
]
