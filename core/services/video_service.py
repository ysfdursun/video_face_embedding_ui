import os
from django.conf import settings
from core.utils.file_utils import get_safe_filename

def save_uploaded_video(video_file, movie_title):
    """Saves the uploaded video file to the media directory."""
    video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
    os.makedirs(video_dir, exist_ok=True)

    safe_title = get_safe_filename(movie_title)
    _, file_ext = os.path.splitext(video_file.name)
    final_filename = f"{safe_title}{file_ext}"

    video_path = os.path.join(video_dir, final_filename)
    with open(video_path, 'wb+') as f:
        for chunk in video_file.chunks():
            f.write(chunk)
    
    return final_filename

def create_movie(title):
    """
    Creates a new movie in the database.
    """
    from core.models import Movie
    from core.utils.file_utils import get_safe_filename
    
    # Check if exists
    if Movie.objects.filter(title=title).exists():
        return None, "Movie already exists"
    
    try:
        movie = Movie.objects.create(title=title)
        return movie, None
    except Exception as e:
        return None, str(e)

def get_uploaded_videos():
    """Returns a list of uploaded video filenames."""
    video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    return [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
