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
    
    try:
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
    
    def stream_with_cleanup():
        import cv2
        import numpy as np
        
        try:
            # Instantiate Robust Recognizer
            recognizer = VideoFaceRecognizer(
                threshold=threshold,
                temporal_buffer_size=buffer_size
            )
            
            # Stream frames
            yield from recognizer.process_stream(temp_path)
            
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
