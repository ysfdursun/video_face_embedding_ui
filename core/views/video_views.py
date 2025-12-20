import os
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse

from core.forms import VideoUploadForm
from core.face_processor import process_and_extract_faces_stream
from core.services.video_service import save_uploaded_video, get_uploaded_videos
from core.services.face_service import count_pending_groups
from core.utils.file_utils import get_safe_filename

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

    return render(request, 'core/home.html', {
        'form': form,
        'videos': videos,
        'pending_groups_count': pending_groups_count,
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
