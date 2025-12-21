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
    """Aktöre fotoğraf ekle (AJAX endpoint)"""
    
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
