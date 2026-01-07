import os
import cv2
import numpy as np
import base64
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from core.models import Actor
from core.face_recognizer import VideoFaceRecognizer
from core.config import Config

@csrf_exempt
def labeller_page(request):
    """Renders the photo labeling tool page."""
    return render(request, 'core/labeller.html')

def get_labeller_recognizer():
    return VideoFaceRecognizer()

@csrf_exempt
def api_analyze_photo(request):
    """
    Analyzes a single uploaded photo with STRICT quality checks.
    Returns: Matches, Quality Stats, and Cropped Face.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)

    img_data = None
    
    # 1. Load Image
    try:
        if 'image' in request.FILES:
            file = request.FILES['image']
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_data = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif 'frame' in request.POST:
            # Base64 handling
            frame_data = request.POST['frame']
            header, encoded = frame_data.split(',', 1)
            data = base64.b64decode(encoded)
            img_data = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            
        if img_data is None:
             return JsonResponse({'success': False, 'error': 'Görüntü okunamadı.'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Dosya okuma hatası: {str(e)}'}, status=400)

    # Custom Settings
    det_thresh = float(request.POST.get('det_thresh', 0.5))
    min_blur = float(request.POST.get('min_blur', Config.MIN_BLUR_SCORE))
    
    recognizer = get_labeller_recognizer()
    
    # Apply Custom Settings
    recognizer.detector.det_thresh = det_thresh
    recognizer.min_blur = min_blur

    # 2. Detect & Select Largest Face
    print(f"DEBUG: Analyzing with Thresh={det_thresh}, MinBlur={min_blur}")
    
    faces = recognizer.detector.get(img_data)
    
    print(f"DEBUG: Detected {len(faces)} faces")
    
    if not faces:
        return JsonResponse({'success': False, 'error': f'Yüz tespit edilemedi. (Eşik: {det_thresh})'}, status=400)

    # Sort largest first
    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    face = faces[0]

    # 3. Quality Checks (Strict Mode)
    # Replicating checks from VideoFaceExtractor
    
    # Size check
    box = face.bbox.astype(int)
    w, h = box[2] - box[0], box[3] - box[1]
    if w < Config.MIN_FACE_SIZE or h < Config.MIN_FACE_SIZE:
        return JsonResponse({'success': False, 'error': f'Yüz çok küçük ({w}x{h}). Min {Config.MIN_FACE_SIZE}px gerekli.'})

    # Blur Check (Laplacian)
    # We need to crop to check blur properly roughly first
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(img_data.shape[1], box[2]), min(img_data.shape[0], box[3])
    face_img = img_data[y1:y2, x1:x2]
    
    if face_img.size == 0:
        return JsonResponse({'success': False, 'error': 'Yüz kırpma hatası.'})

    # Blur Check
    blur_score = FaceQualityScorer.calculate_blur_score(face_img)
    if blur_score < min_blur:
         return JsonResponse({'success': False, 'error': f'Görüntü çok bulanık. (Skor: {blur_score:.1f}, Min: {min_blur})'}, status=400)

    # 4. Extract Embedding & Align
    # This aligns the face to 112x112
    # Note: extraction method inside recognizer might trigger its own checks, but we did pre-checks.
    # We use the raw embedding extraction to be safe.
    
    # We'll use a helper or the recognizer's method
    try:
        # Assuming extract_valid_embedding logic or similar
        # For this tool, we trust recognizer.extract_embedding handles alignment
        embedding = recognizer.extract_embedding(face, img_data) 
        if embedding is None:
             return JsonResponse({'success': False, 'error': 'Yüz hizalama veya özellik çıkarma başarısız.'}, status=500)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'İşleme hatası: {str(e)}'}, status=500)

    # 5. Find Matches
    matches = recognizer.find_nearest_neighbors(embedding, k=5)
    
    # Enrich matches with profile images (optional, if we want visuals in list)
    # Matches structure: [{'name': 'Brad Pitt', 'score': 0.85}, ...]
    # Convert score to percentage
    formatted_matches = []
    for m in matches:
        score_pct = int(m['score'] * 100)
        formatted_matches.append({
            'name': m['name'],
            'score': score_pct
        })

    # 6. Prepare Aligned Face for display
    # We need to re-run alignment to get the crisp image for user preview
    # Using INSIGHTFACE helper for standard 112x112 crop
    from insightface.utils import face_align
    aligned_face = face_align.norm_crop(img_data, landmark=face.kps)
    
    # Encode aligned face
    _, buffer = cv2.imencode('.jpg', aligned_face)
    aligned_b64 = base64.b64encode(buffer).decode('utf-8')
    aligned_data_url = f"data:image/jpeg;base64,{aligned_b64}"

    return JsonResponse({
        'success': True,
        'quality': {
            'blur_score': int(blur_score),
            'size': f"{w}x{h}"
        },
        'cropped_image': aligned_data_url,
        'matches': formatted_matches
    })

@csrf_exempt
def api_enroll_face(request):
    """
    Enrolls the face into the labeled dataset.
    Receives: 'name' and 'image' (file).
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
        
    name = request.POST.get('name', '').strip()
    if not name:
        return JsonResponse({'success': False, 'error': 'İsim gerekli.'}, status=400)
        
    try:
        from core.services.enrollment_service import enroll_uploaded_image
        
        if 'image' not in request.FILES:
             return JsonResponse({'success': False, 'error': 'Fotoğraf dosyası eksik.'}, status=400)
             
        file = request.FILES['image']
        
        # Use centralized service that handles File + DB + PKL
        result = enroll_uploaded_image(name, file)
        
        if result['success']:
            warning = result.get('warning', '')
            if warning:
                return JsonResponse({'success': True, 'message': 'Kaydedildi (Uyarı: ' + warning + ')'})
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'success': False, 'error': result.get('error', 'Kayıt başarısız')})
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Kayıt hatası: {str(e)}'}, status=500)
