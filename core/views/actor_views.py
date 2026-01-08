from django.shortcuts import render, redirect
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import JsonResponse
from django.conf import settings

from core.services import face_service, actor_service

def actors_dashboard(request):
    """
    ✅ OPTIMIZED DASHBOARD: Hızlı yükleme ile kalite analizi
    """
    search_query = request.GET.get('search', '').strip().lower().replace(' ', '_')
    page = request.GET.get('page', 1)
    
    # Service call
    actors_data = actor_service.get_actors_data(search_query)
    
    # ⚡ Pagination: 15 aktör/sayfa
    paginator = Paginator(actors_data, 15)
    
    try:
        actors_page = paginator.page(page)
    except PageNotAnInteger:
        actors_page = paginator.page(1)
    except EmptyPage:
        actors_page = paginator.page(paginator.num_pages)
    
    # Toplam istatistikler (sadece görüntülenen sayfadaki)
    total_actors = len(actors_data)
    total_photos = sum(a['photo_count'] for a in actors_data)
    
    context = {
        'actors': actors_page.object_list,
        'actors_page': actors_page,
        'total_actors': total_actors,
        'total_photos': total_photos,
        'search_query': search_query,
        'labeled_dir': settings.MEDIA_ROOT + '/labeled_faces',  # AJAX için
    }
    
    return render(request, 'core/actors_dashboard.html', context)

def actor_detail(request, actor_name):
    """
    ✅ AKTÖR DETAY SAYFASI: Aktörün tüm fotoğraflarını göster + kalite analizi
    """
    context = actor_service.get_actor_details(actor_name)
    
    if not context:
        return redirect('core:actors_dashboard')
    
    return render(request, 'core/actor_detail.html', context)

def upload_photo(request):
    """Aktöre fotoğraf ekle (AJAX endpoint) for the gallery"""
    
    if request.method == 'POST' and request.FILES.get('photo'):
        actor_name = request.POST.get('actor_name')
        photo_file = request.FILES['photo']
        
        if not actor_name:
            return JsonResponse({'success': False, 'message': 'Aktör adı eksik'})
        
        try:
            filename = face_service.save_actor_photo(actor_name, photo_file)
            return JsonResponse({
                'success': True, 
                'message': 'Fotoğraf başarıyla yüklendi',
                'filename': filename
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'Yükleme hatası: {str(e)}'})
    
    return JsonResponse({'success': False, 'message': 'Geçersiz istek'})

def update_profile_photo(request):
    """Update actor profile photo in Selected_Profiles"""
    if request.method == 'POST' and request.FILES.get('photo'):
        actor_name = request.POST.get('actor_name')
        photo_file = request.FILES['photo']
        
        try:
            filename = actor_service.save_profile_photo(actor_name, photo_file)
            return JsonResponse({
                'success': True,
                'message': 'Profil fotoğrafı güncellendi',
                'filename': filename,
                'url': f"{settings.MEDIA_URL}Selected_Profiles/{actor_name}/{filename}"
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'Hata: {str(e)}'})
    return JsonResponse({'success': False, 'message': 'Geçersiz istek'})

def remove_profile_photo(request):
    """Remove actor profile photo from Selected_Profiles"""
    if request.method == 'POST':
        actor_name = request.POST.get('actor_name')
        
        try:
            success = actor_service.delete_profile_photo(actor_name)
            if success:
                return JsonResponse({'success': True, 'message': 'Profil fotoğrafı kaldırıldı'})
            else:
                return JsonResponse({'success': False, 'message': 'Fotoğraf bulunamadı'})
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'Hata: {str(e)}'})
    return JsonResponse({'success': False, 'message': 'Geçersiz istek'})

def toggle_embedding(request):
    """Toggle embedding status for a specific photo"""
    if request.method == 'POST':
        actor_name = request.POST.get('actor_name')
        photo_path = request.POST.get('photo_path')
        
        if not actor_name or not photo_path:
            return JsonResponse({'success': False, 'message': 'Eksik parametre'})
            
        try:
            from core.services.embedding_service import embedding_service
            
            # photo_path comes like 'labeled_faces/Name/img.jpg', normalize relative to MEDIA_ROOT
            # Actually, the service expects 'labeled_faces/Name/img.jpg' if that's what we pass
             
            success, new_status, message = embedding_service.toggle_embedding(actor_name, photo_path)
            
            return JsonResponse({
                'success': success,
                'is_embedded': new_status,
                'message': message
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'message': f'Hata: {str(e)}'})
            
    return JsonResponse({'success': False, 'message': 'Geçersiz istek'})
