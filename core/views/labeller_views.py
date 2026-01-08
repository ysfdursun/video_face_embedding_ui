import os
import cv2
import numpy as np
import base64
import json
import time
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

# Global cache for the heavyweight recognizer (DB & Matrix)
_LABELLER_RECOGNIZER = None

def get_labeller_recognizer():
    global _LABELLER_RECOGNIZER
    if _LABELLER_RECOGNIZER is None:
        print("⚡ Initializing Labeller Recognizer (Loading DB)...")
        _LABELLER_RECOGNIZER = VideoFaceRecognizer()
    return _LABELLER_RECOGNIZER

@csrf_exempt
def api_analyze_photo(request):
    """
    Analyzes a single uploaded photo.
    Aligned with Lookalike logic for robustness.
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

    # Custom Settings (Optional override)
    det_thresh = float(request.POST.get('det_thresh', 0.5))
    min_blur = float(request.POST.get('min_blur', Config.MIN_BLUR_SCORE))
    min_face_size = int(request.POST.get('min_face_size', Config.MIN_FACE_SIZE))
    
    # 2. Hybrid Detection Strategy
    recognizer = get_labeller_recognizer()
    
    # Strategy A: Use Global Recognizer (Fast, Cached)
    # Temporarily override threshold
    original_thresh = recognizer.detector.det_thresh
    recognizer.detector.det_thresh = det_thresh
    
    print(f"DEBUG: Strategy A (Global) - Analyzing with Thresh={det_thresh}")
    faces = recognizer.detector.get(img_data)
    recognizer.detector.det_thresh = original_thresh # Restore
    
    # Strategy B: Fallback to Fresh Detector (If Global failed)
    # We try this even if image is small, just in case global detector has state issues
    if not faces:
        h, w = img_data.shape[:2]
        print(f"DEBUG: Strategy A failed. Attempting Strategy B (Fresh Detector) for {w}x{h} image...")
        
        # Determine size
        # Force 640x640 at minimum if small, or dynamic if large
        # This ensures small images are upscaled if needed by InsightFace
        det_size = (640, 640)
        if w > 640 or h > 640:
            max_dim = max(w, h)
            target_dim = min(max_dim, 1280)
            target_dim = (target_dim // 32) * 32
            det_size = (target_dim, target_dim)
        
        print(f"DEBUG: Strategy B - Analyzing with Size={det_size}, Thresh={det_thresh}")
            
        try:
            # DIRECT INSTANTIATION (Bypassing model_loader to match working script)
            from insightface.app import FaceAnalysis
            # Use 'buffalo_sc' directly as we know it works
            dyn_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
            dyn_app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
            faces = dyn_app.get(img_data)
        except Exception as e:
            print(f"DEBUG: Strategy B Error: {e}")

    print(f"DEBUG: Final Detection Count: {len(faces)}")
    
    if not faces:
        # SAVE DEBUG IMAGE for inspection
        try:
            debug_dir = os.path.join(settings.MEDIA_ROOT, 'debug_failed_detections')
            os.makedirs(debug_dir, exist_ok=True)
            ts = int(time.time())
            cv2.imwrite(os.path.join(debug_dir, f"failed_{ts}.jpg"), img_data)
            print(f"DEBUG: Saved failed image to debug_failed_detections/failed_{ts}.jpg")
        except:
            pass
            
        return JsonResponse({'success': False, 'error': f'Yüz tespit edilemedi. (Eşik: {det_thresh}). Hassasiyeti arttırmayı (değeri düşürmeyi) deneyin.'}, status=400)

    # Sort largest first
    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    face = faces[0]

    # 3. Quality Checks 
    # Size check
    box = face.bbox.astype(int)
    w_box, h_box = box[2] - box[0], box[3] - box[1]
    
    if w_box < min_face_size or h_box < min_face_size:
        return JsonResponse({'success': False, 'error': f'Yüz çok küçük ({w_box}x{h_box}). Min {min_face_size}px gerekli.'}, status=400)

    # Blur Check
    # Crop for blur check
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(img_data.shape[1], box[2]), min(img_data.shape[0], box[3])
    face_img = img_data[y1:y2, x1:x2]
    
    if face_img.size == 0:
        return JsonResponse({'success': False, 'error': 'Yüz kırpma hatası.'}, status=400)

    # Blur Check (Raw Laplacian Variance)
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if blur_score < min_blur:
         return JsonResponse({'success': False, 'error': f'Görüntü çok bulanık. (Skor: {blur_score:.1f}, Min: {min_blur})'}, status=400)

    # 4. Extract Embedding
    try:
        # Note: If face was detected by dyn_detector, extracting embedding using global recognizer is still valid
        # as long as we pass the raw img_data and the face object (which contains bbox/kps).
        embedding = recognizer.extract_embedding(face, img_data) 
        if embedding is None:
             return JsonResponse({'success': False, 'error': 'Yüz hizalama veya özellik çıkarma başarısız.'}, status=500)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'İşleme hatası: {str(e)}'}, status=500)

    # 5. Find Matches
    matches = recognizer.find_nearest_neighbors(embedding, k=5)
    
    formatted_matches = []
    for m in matches:
        score_pct = int(m['score'] * 100)
        formatted_matches.append({
            'name': m['name'],
            'score': score_pct
        })

    # 6. Prepare Aligned Face
    from insightface.utils import face_align
    aligned_face = face_align.norm_crop(img_data, landmark=face.kps)
    _, buffer = cv2.imencode('.jpg', aligned_face)
    aligned_b64 = base64.b64encode(buffer).decode('utf-8')
    aligned_data_url = f"data:image/jpeg;base64,{aligned_b64}"

    return JsonResponse({
        'success': True,
        'quality': {
            'blur_score': int(blur_score),
            'size': f"{w_box}x{h_box}"
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
