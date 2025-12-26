from django.shortcuts import render, redirect
from django.contrib import messages
from core.models import FaceRecognitionSettings

def settings_view(request):
    settings_obj = FaceRecognitionSettings.get_settings()
    
    if request.method == 'POST':
        try:
            settings_obj.detection_threshold = float(request.POST.get('detection_threshold', 0.5))
            settings_obj.grouping_threshold = float(request.POST.get('grouping_threshold', 0.45))
            settings_obj.min_face_size = int(request.POST.get('min_face_size', 80))
            settings_obj.frame_skip_extract = int(request.POST.get('frame_skip_extract', 10))
            settings_obj.frame_skip_group = int(request.POST.get('frame_skip_group', 5))
            settings_obj.gpu_enabled = request.POST.get('gpu_enabled') == 'on'
            
            # Quality & Classification settings
            settings_obj.quality_threshold = float(request.POST.get('quality_threshold', 0.50))
            settings_obj.redundancy_threshold = float(request.POST.get('redundancy_threshold', 0.85))
            
            settings_obj.save()
            messages.success(request, 'Ayarlar başarıyla güncellendi.')
        except ValueError:
            messages.error(request, 'Geçersiz değerler girildi.')
        
        return redirect('core:settings')
        
    return render(request, 'core/settings.html', {'settings': settings_obj})
