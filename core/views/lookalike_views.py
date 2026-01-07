# -*- coding: utf-8 -*-
import base64
import numpy as np
import cv2
import os
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from core.face_recognizer import VideoFaceRecognizer
from core.config import Config
from core.model_loader import load_recognizer, load_detector

# Initialize a global recognizer for lookalike (Lazy load)
# We use VideoFaceRecognizer but only for its DB search capability.
# For detection/extraction on single image, we can use it or raw components.
# Reuse VideoFaceRecognizer for simplicity of dependency injection.
LOOKALIKE_RECOGNIZER = None

def get_lookalike_recognizer():
    global LOOKALIKE_RECOGNIZER
    if LOOKALIKE_RECOGNIZER is None:
        # Init with defaults (threshold doesn't matter much for Top-K)
        LOOKALIKE_RECOGNIZER = VideoFaceRecognizer(threshold=0.2)
    return LOOKALIKE_RECOGNIZER

def lookalike_page(request):
    """Render the Lookalike page."""
    return render(request, 'core/lookalike.html')

@csrf_exempt
def api_analyze_lookalike(request):
    """
    Analyze uploaded image or webcam frame.
    Returns Top-5 matches.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'}, status=405)
        
    try:
        # Get Image Data
        img_data = None
        
        # 1. File Upload
        if 'image' in request.FILES:
            file = request.FILES['image']
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_data = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
        # 2. Webcam Frame (Base64)
        elif 'frame' in request.POST:
            frame_b64 = request.POST.get('frame')
                
            if frame_b64:
                # Remove header if present (data:image/jpeg;base64,...)
                if ',' in frame_b64:
                    frame_b64 = frame_b64.split(',')[1]
                
                try:
                    img_bytes = base64.b64decode(frame_b64)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    img_data = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except Exception as e:
                    print(f"Base64 decode error: {e}")
                    return JsonResponse({'success': False, 'error': 'Invalid image data'}, status=400)
                
        if img_data is None:
             return JsonResponse({'success': False, 'error': 'No image data provided'}, status=400)
             
        # Process Image
        recognizer = get_lookalike_recognizer()
        
        # Detect Face
        # Use the recognizer's detector
        faces = recognizer.detector.get(img_data)
        
        if not faces:
            return JsonResponse({'success': False, 'error': 'No face detected. Please try again with better lighting.'}, status=400)
            
        # Take largest face
        faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        face = faces[0]
        
        # Extract Embedding
        embedding = recognizer.extract_embedding(face, img_data)
        if embedding is None:
             return JsonResponse({'success': False, 'error': 'Could not extract face features.'}, status=500)
             
        # Find Neighbors
        matches = recognizer.find_nearest_neighbors(embedding, k=5)
        
        # Enrich Matches with Profile Images
        enriched_matches = []
        for match in matches:
            name = match['name']
            score = match['score']
            
            # Find image
            image_url = None
            
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
                        
            enriched_matches.append({
                'name': name,
                'score': round(score * 100, 1), # Percentage
                'image': image_url
            })
            
        return JsonResponse({'success': True, 'matches': enriched_matches})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
