import os
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse

from core.forms import VideoUploadForm
from core.face_processor import process_and_extract_faces_stream
from core.services.video_service import save_uploaded_video, get_uploaded_videos
from core.services.face_service import count_pending_groups, count_pending_faces
from core.utils.file_utils import get_safe_filename
from core.models import Actor, Movie

def welcome(request):
    """Landing page with upload/label options."""
    return render(request, 'core/welcome.html')

def home(request):
    """
    Main page. Handles video upload and lists uploaded videos and pending unlabeled faces.
    """
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = request.FILES['video_file']
            movie = form.cleaned_data['movie']
            final_filename = save_uploaded_video(video_file, movie.title)
            return redirect('core:processing_page', movie_filename=final_filename)
    else:
        form = VideoUploadForm()

    videos = get_uploaded_videos()
    pending_groups_count = count_pending_groups()
    pending_faces = count_pending_faces()
    
    # --- Statistics for Dashboard ---
    total_actors = Actor.objects.count()
    total_movies = Movie.objects.count()
    
    # Count total labeled faces (approximate via filesystem walk or cache)
    # Simple recursive count for now (limit depth to avoid hangs on huge datasets, but labeled_faces is flat-ish)
    total_labeled_faces = 0
    labeled_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces')
    if os.path.exists(labeled_dir):
        for _, _, files in os.walk(labeled_dir):
            total_labeled_faces += len([f for f in files if f.endswith('.jpg')])

    today_stats = {
        'total_labeled_faces_safe': float(total_labeled_faces),
        'pending_faces_safe': float(pending_faces),
    }

    # Calculate Rates
    total_faces_all = total_labeled_faces + pending_faces
    if total_faces_all > 0:
        identification_rate = int((total_labeled_faces / total_faces_all) * 100)
    else:
        identification_rate = 0
        
    # Dataset Quality (Simplified: 100% if everything is labeled, else penalize)
    if pending_faces == 0:
        quality_rate = 100
    else:
        quality_rate = max(0, 100 - int((pending_faces / (total_faces_all + 1)) * 100))

    return render(request, 'core/home.html', {
        'form': form,
        'videos': videos,
        'pending_groups_count': pending_groups_count,
        'stats': {
             'total_actors': total_actors,
             'total_movies': total_movies,
             'total_videos': len(videos),
             'total_labeled_faces': total_labeled_faces,
             'pending_faces': pending_faces,
             'identification_rate': identification_rate,
             'quality_rate': quality_rate,
        }
    })

def processing_page(request, movie_filename):
    """Video processing screen that shows the stream."""
    movie_title = os.path.splitext(movie_filename)[0]
    return render(request, 'core/processing.html', {'movie_title': movie_title, 'movie_filename': movie_filename})

def stream_video_processing(request, movie_filename):
    """Runs the processing generator and starts background grouping."""
    video_path = os.path.join(settings.MEDIA_ROOT, 'videos', movie_filename)
    movie_title = os.path.splitext(movie_filename)[0]

    try:
        # UNIFIED PIPELINE: Extract + Group aynı anda (1 pass)
        stream = process_and_extract_faces_stream(video_path, movie_title, group_faces=True)
        return StreamingHttpResponse(stream, content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Stream error: {e}")
        import traceback
        traceback.print_exc()
        return redirect('core:home')

def stream_video_recognition(request, movie_filename):
    """Runs the recognition stream."""
    video_path = os.path.join(settings.MEDIA_ROOT, 'videos', movie_filename)
    
    try:
        from core.face_recognizer import VideoFaceRecognizer
        # Use default settings for existing videos since this view doesn't take params
        recognizer = VideoFaceRecognizer(threshold=0.35, temporal_buffer_size=10)
        stream = recognizer.process_stream(video_path)
        return StreamingHttpResponse(stream, content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Recognition stream error: {e}")
        return redirect('core:home')
