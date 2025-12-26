# -*- coding: utf-8 -*-
"""
Recognition Views - Video Face Recognition Page
"""
import os
import tempfile
import uuid
from django.conf import settings
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from core.services.recognition_service import (
    process_recognition_stream,
    get_identities_count
)

# Temp file storage (in-memory reference for cleanup)
_temp_files = {}

def recognition_page(request):
    """Render the recognition page."""
    return render(request, 'core/recognition.html', {
        'identities_count': get_identities_count()
    })

@csrf_exempt
def recognition_upload(request):
    """
    Handle video upload. Save to temp file and return a session ID.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    video_file = request.FILES.get('video')
    if not video_file:
        return JsonResponse({'error': 'No video file'}, status=400)
    
    # Generate session ID
    session_id = str(uuid.uuid4())[:8]
    
    # Save to temp file
    suffix = os.path.splitext(video_file.name)[1] or '.mp4'
    temp_dir = os.path.join(settings.BASE_DIR, 'temp_recognition')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, f"{session_id}{suffix}")
    
    print(f"📤 Uploading video to: {temp_path}")
    
    with open(temp_path, 'wb') as f:
        for chunk in video_file.chunks():
            f.write(chunk)
    
    file_size = os.path.getsize(temp_path)
    print(f"✅ Upload complete: {file_size} bytes")
    
    return JsonResponse({
        'success': True,
        'session_id': session_id,
        'filename': video_file.name
    })

def recognition_stream(request, session_id):
    """
    Stream recognition for uploaded video by session ID.
    """
    print(f"🎬 Stream request for session: {session_id}")
    
    # Find temp file by session_id (scan temp directory)
    temp_dir = os.path.join(settings.BASE_DIR, 'temp_recognition')
    temp_path = None
    
    if os.path.exists(temp_dir):
        for fname in os.listdir(temp_dir):
            if fname.startswith(session_id):
                temp_path = os.path.join(temp_dir, fname)
                break
    
    if not temp_path or not os.path.exists(temp_path):
        print(f"❌ Session not found: {session_id}")
        print(f"   Temp dir contents: {os.listdir(temp_dir) if os.path.exists(temp_dir) else 'N/A'}")
        return JsonResponse({'error': 'Session not found'}, status=404)
    
    print(f"✅ Found temp file: {temp_path}")
    
    # Get settings from query params
    threshold = float(request.GET.get('threshold', 0.27))
    min_face_size = int(request.GET.get('min_face_size', 30))
    frame_skip = int(request.GET.get('frame_skip', 5))
    
    def stream_with_cleanup():
        try:
            for frame in process_recognition_stream(
                temp_path,
                threshold=threshold,
                min_face_size=min_face_size,
                frame_skip=frame_skip
            ):
                yield frame
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

def recognition_stream_existing(request, video_filename):
    """Stream recognition for an existing video in media/videos."""
    video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_filename)
    
    if not os.path.exists(video_path):
        return JsonResponse({'error': 'Video not found'}, status=404)
    
    # Get settings from query params
    threshold = float(request.GET.get('threshold', 0.27))
    min_face_size = int(request.GET.get('min_face_size', 30))
    frame_skip = int(request.GET.get('frame_skip', 5))
    
    stream = process_recognition_stream(
        video_path,
        threshold=threshold,
        min_face_size=min_face_size,
        frame_skip=frame_skip
    )
    
    return StreamingHttpResponse(
        stream,
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
