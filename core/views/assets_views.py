import os
import io
from PIL import Image
from django.conf import settings
from django.http import FileResponse, JsonResponse

def serve_optimized_image(request, image_path):
    """
    ✅ PERFORMANS OPTİMİZASYONU: Resim cache'i ve compression ile serve et
    """
    if not image_path:
        return JsonResponse({'error': 'Image path required'}, status=400)
    
    # Security check: path traversal'ı engelle
    if '..' in image_path or image_path.startswith('/'):
        return JsonResponse({'error': 'Invalid path'}, status=403)
    
    # Dosya yolunu oluştur
    full_path = os.path.join(settings.MEDIA_ROOT, image_path)
    
    # Security: Dosyanın MEDIA_ROOT içinde olduğundan emin ol
    if not os.path.abspath(full_path).startswith(os.path.abspath(settings.MEDIA_ROOT)):
        return JsonResponse({'error': 'Access denied'}, status=403)
    
    if not os.path.isfile(full_path):
        return JsonResponse({'error': 'Image not found'}, status=404)
    
    try:
        # Query parameters'dan width ve quality al (varsayılan değerler)
        width = int(request.GET.get('w', 300))  # Default: 300px
        quality = int(request.GET.get('q', 75))  # Default: 75%
        
        # PIL ile resmi aç ve optimize et
        img = Image.open(full_path)
        
        # RGBA'yı RGB'ye dönüştür (JPEG compatibility)
        if img.mode in ('RGBA', 'LA'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = rgb_img
        
        # Resize: aspect ratio'yu koru
        img.thumbnail((width, width), Image.Resampling.LANCZOS)
        
        # BytesIO buffer'a kaydet
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        output.seek(0)
        
        # HTTP response
        response = FileResponse(output, content_type='image/jpeg')
        response['Cache-Control'] = 'public, max-age=2592000'  # 30 günlük cache
        return response
    
    except Exception as e:
        print(f"[OPTIMIZE_IMAGE] Error: {e}")
        # Fallback: orijinal resmi serve et
        return FileResponse(open(full_path, 'rb'), content_type='image/jpeg')
