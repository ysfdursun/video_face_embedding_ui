# -*- coding: utf-8 -*-
"""
Recognition Views - Video Face Recognition Page
"""
import os
import uuid
from django.conf import settings
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Import our new Robust Recognizer
from core.face_recognizer import VideoFaceRecognizer
from core.models import FaceGroup

# Helper to get identity count
def get_identities_count():
    # Count groups that have a name and a representative embedding
    return FaceGroup.objects.exclude(name='').count()

def recognition_page(request):
    """Render the premium recognition dashboard."""
    return render(request, 'core/recognition.html', {
        'identities_count': get_identities_count()
    })

@csrf_exempt
def recognition_upload(request):
    """
    Handle video upload. Save to temp file and return a session ID.
    The temp file will be deleted after streaming logic cleans it up.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    video_file = request.FILES.get('video')
    movie_id = request.POST.get('movie_id') # Optional movie context
    
    if not video_file:
        return JsonResponse({'error': 'No video file'}, status=400)
    
    # Generate session ID
    session_id = str(uuid.uuid4())[:8]
    
    # Save to temp file
    suffix = os.path.splitext(video_file.name)[1] or '.mp4'
    temp_dir = os.path.join(settings.BASE_DIR, 'temp_recognition')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, f"{session_id}{suffix}")
    
    print(f"📤 Uploading video to: {temp_path} (Context Movie ID: {movie_id})")
    
    try:
        with open(temp_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)
        
        file_size = os.path.getsize(temp_path)
        print(f"✅ Upload complete: {file_size} bytes")
        
        return JsonResponse({
            'success': True,
            'session_id': session_id,
            'filename': video_file.name,
            'movie_id': movie_id # Pass it back so frontend can pass it to stream
        })
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

def recognition_stream(request, session_id):
    """
    Stream recognition for uploaded video by session ID.
    Uses VideoFaceRecognizer with Temporal Smoothing.
    """
    print(f"🎬 Stream request for session: {session_id}")
    
    # Find temp file by session_id
    temp_dir = os.path.join(settings.BASE_DIR, 'temp_recognition')
    temp_path = None
    
    if os.path.exists(temp_dir):
        for fname in os.listdir(temp_dir):
            if fname.startswith(session_id):
                temp_path = os.path.join(temp_dir, fname)
                break
    
    if not temp_path or not os.path.exists(temp_path):
        return JsonResponse({'error': 'Session not found'}, status=404)
    
    # Get settings from query params
    threshold = float(request.GET.get('threshold', 0.27))
    buffer_size = int(request.GET.get('buffer_size', 10)) # Temporal buffer size
    movie_id = request.GET.get('movie_id') # Optional movie ID from upload step
    
    def stream_with_cleanup():
        import cv2
        import numpy as np
        
        target_identities = None
        if movie_id and movie_id != 'null':
            from core.models import MovieCast
            # Fetch cast names
            cast_names = list(MovieCast.objects.filter(movie_id=movie_id).values_list('actor__name', flat=True))
            if cast_names:
                target_identities = cast_names
                print(f"🎯 Movie Context Loaded: {len(cast_names)} cast members found.")
        
        try:
            # New Manual Settings
            min_blur = float(request.GET.get('min_blur', 0)) # Default 0 (Disabled)
            min_face_size = int(request.GET.get('min_face_size', 20)) # Default 20
            
            # Instantiate Robust Recognizer
            recognizer = VideoFaceRecognizer(
                threshold=threshold,
                temporal_buffer_size=buffer_size,
                target_identities=target_identities,
                min_blur=min_blur,
                min_face_size=min_face_size
            )
            
            # Stream frames
            yield from recognizer.process_stream(temp_path, session_id=session_id)
            
        except Exception as e:
            print(f"Stream Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Generate Error Frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Draw multiple lines of text
            err_msg = str(e)
            y0, dy = 100, 30
            
            cv2.putText(error_frame, "SYSTEM ERROR:", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Split long messages
            for i, line in enumerate(err_msg.split(',')):
                y = y0 + i * dy
                if y > 450: break
                cv2.putText(error_frame, line.strip(), (50, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                # Yield error frame multiple times to ensure it's displayed
                frame_data = (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                for _ in range(10):
                    yield frame_data
                    
        finally:
            # Cleanup temp file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"🗑️ Cleaned up: {temp_path}")
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")
    
    return StreamingHttpResponse(
        stream_with_cleanup(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def api_recognition_status(request):
    """
    API to poll recognition status and stats.
    """
    session_id = request.GET.get('session_id')
    if not session_id:
        return JsonResponse({'error': 'No session_id'}, status=400)
        
    from django.core.cache import cache
    data = cache.get(f"rec_stats_{session_id}")
    
    if data:
        # Enrich with profile photos
        stats_dict = data.get('stats', {})
        enriched_actors = []
        
        for name, count in stats_dict.items():
            image_url = None
            
            if name == "Unknown":
                image_url = None # Placeholder handled by frontend
            elif name.startswith("Misafir"):
                # 3. Check for specific session guest image
                guest_path = os.path.join(settings.MEDIA_ROOT, 'temp_faces', session_id, f"{name}.jpg")
                if os.path.exists(guest_path):
                     image_url = f"{settings.MEDIA_URL}temp_faces/{session_id}/{name}.jpg"
            else:
                # 1. Try Selected_Profiles
                profile_dir = os.path.join(settings.MEDIA_ROOT, 'Selected_Profiles', name)
                if os.path.isdir(profile_dir):
                    files = [f for f in os.listdir(profile_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    if files:
                        image_url = f"{settings.MEDIA_URL}Selected_Profiles/{name}/{files[0]}"
                
                # 2. Fallback to labeled_faces
                if not image_url:
                    labeled_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces', name)
                    if os.path.isdir(labeled_dir):
                        files = [f for f in os.listdir(labeled_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                        if files:
                            image_url = f"{settings.MEDIA_URL}labeled_faces/{name}/{files[0]}"

            enriched_actors.append({
                'name': name,
                'count': count,
                'image': image_url,
                'is_guest': name.startswith("Misafir"),
                'is_unknown': name == "Unknown"
            })
            
        # Custom Sorting Logic
        # 1. Known Actors (by count desc)
        # 2. Guests (by count desc)
        # 3. Unknown
        enriched_actors.sort(key=lambda x: (
            x['name'] == "Unknown", # Unknown last (True=1, False=0)
            x['is_guest'],          # Guest second to last
            -x['count']             # Higher count first
        ))
            
        data['actors'] = enriched_actors
        return JsonResponse(data)
    else:
        return JsonResponse({'status': 'waiting', 'stats': {}, 'actors': []})

def api_search_movies(request):
    """
    API for movie autocomplete.
    """
    query = request.GET.get('q', '').strip()
    if len(query) < 2:
        return JsonResponse([], safe=False)
        
    from core.models import Movie
    movies = Movie.objects.filter(title__icontains=query)[:10]
    
    results = [{'id': m.id, 'title': m.title} for m in movies]
    return JsonResponse(results, safe=False)
