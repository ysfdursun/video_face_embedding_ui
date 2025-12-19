# -*- coding: utf-8 -*-
from django.http import StreamingHttpResponse, FileResponse, JsonResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.urls import reverse
from django.core.cache import cache
import os
import shutil
import cv2
import numpy as np
from .forms import VideoUploadForm
from .models import Movie, MovieCast
from .face_processor import process_and_extract_faces_stream


def get_safe_filename(name):
    """Remove special chars and spaces to build a safe filename/folder name."""
    return "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).rstrip().replace(' ', '_')


def welcome(request):
    """Landing page with upload/label options."""
    return render(request, 'core/welcome.html')


def home(request):
    """
    Main page. Handles video upload and lists uploaded videos and pending unlabeled faces.
    """
    video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
    os.makedirs(video_dir, exist_ok=True)

    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = request.FILES['video_file']
            movie = form.cleaned_data['movie']
            movie_title = movie.title

            safe_title = get_safe_filename(movie_title)
            _, file_ext = os.path.splitext(video_file.name)
            final_filename = f"{safe_title}{file_ext}"

            video_path = os.path.join(video_dir, final_filename)
            with open(video_path, 'wb+') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)

            return redirect('core:processing_page', movie_filename=final_filename)
    else:
        form = VideoUploadForm()

    videos = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]

    # Grouped faces'ten etiketlenecek grupları say
    grouped_faces_dir = os.path.join(settings.MEDIA_ROOT, 'grouped_faces')
    pending_groups_count = 0
    if os.path.exists(grouped_faces_dir):
        for movie_folder in os.listdir(grouped_faces_dir):
            movie_path = os.path.join(grouped_faces_dir, movie_folder)
            if os.path.isdir(movie_path):
                pending_groups_count += len([f for f in os.listdir(movie_path) if f.startswith('celebrity_') and os.path.isdir(os.path.join(movie_path, f))])

    return render(request, 'core/home.html', {
        'form': form,
        'videos': videos,
        'pending_groups_count': pending_groups_count,
    })


def processing_page(request, movie_filename):
    """Video processing screen that shows the stream."""
    movie_title = os.path.splitext(movie_filename)[0]
    return render(request, 'core/processing.html', {'movie_title': movie_title, 'movie_filename': movie_filename})


def stream_video_processing(request, movie_filename):
    """Runs the processing generator and starts background grouping."""
    video_path = os.path.join(settings.MEDIA_ROOT, 'videos', movie_filename)
    movie_title = os.path.splitext(movie_filename)[0]

    try:
        # UNIFIED PIPELINE: Extract + Group aynı anda (1 pass)
        stream = process_and_extract_faces_stream(video_path, movie_title, group_faces=True)
        return StreamingHttpResponse(stream, content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Stream error: {e}")
        import traceback
        traceback.print_exc()
        return redirect('core:home')


def list_unlabeled_faces(request):
    """List all unlabeled faces as a gallery."""
    unlabeled_dir = os.path.join(settings.MEDIA_ROOT, 'unlabeled_faces')
    os.makedirs(unlabeled_dir, exist_ok=True)

    unlabeled_faces = []
    for movie_folder in sorted(os.listdir(unlabeled_dir)):
        movie_path = os.path.join(unlabeled_dir, movie_folder)
        if os.path.isdir(movie_path):
            for filename in sorted(os.listdir(movie_path)):
                if filename.endswith('.jpg'):
                    unlabeled_faces.append({
                        'path': os.path.join('unlabeled_faces', movie_folder, filename),
                        'movie_title': movie_folder
                    })

    context = {
        'unlabeled_faces': unlabeled_faces,
        'unlabeled_faces_count': len(unlabeled_faces),
    }
    return render(request, 'core/list_unlabeled.html', context)


def delete_single_face(request):
    """AJAX ile tek bir fotoğrafı sil"""
    from django.http import JsonResponse
    
    if request.method == 'POST':
        face_path = request.POST.get('face_path')
        if face_path:
            # Forward slash'i sistem path'ine çevir
            face_path_parts = face_path.split('/')
            full_path = os.path.join(settings.MEDIA_ROOT, *face_path_parts)
            
            print(f"[DELETE] Received: {face_path}")
            print(f"[DELETE] Full path: {full_path}")
            print(f"[DELETE] Exists: {os.path.isfile(full_path)}")
            
            if os.path.isfile(full_path):
                try:
                    os.remove(full_path)
                    print(f"[DELETE] ✓ SUCCESS")
                    return JsonResponse({'success': True})
                except Exception as e:
                    print(f"[DELETE] ✗ ERROR: {e}")
                    return JsonResponse({'success': False, 'message': str(e)})
            else:
                print(f"[DELETE] ✗ FILE NOT FOUND")
                return JsonResponse({'success': False, 'message': 'Dosya bulunamadı'})
        return JsonResponse({'success': False, 'message': 'face_path eksik'})
    return JsonResponse({'success': False, 'message': 'Geçersiz istek'})


def upload_photo(request):
    """Aktöre fotoğraf ekle (AJAX endpoint)"""
    from django.http import JsonResponse
    
    if request.method == 'POST' and request.FILES.get('photo'):
        actor_name = request.POST.get('actor_name')
        photo_file = request.FILES['photo']
        
        if not actor_name:
            return JsonResponse({'success': False, 'message': 'Aktör adı eksik'})
        
        # Aktör klasörünü oluştur
        actor_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces', actor_name)
        os.makedirs(actor_dir, exist_ok=True)
        
        # Dosya adını oluştur (timestamp ile unique yapılması için)
        import time
        ext = os.path.splitext(photo_file.name)[1]
        filename = f"{int(time.time() * 1000)}{ext}"
        file_path = os.path.join(actor_dir, filename)
        
        try:
            # Dosyayı kaydet
            with open(file_path, 'wb+') as destination:
                for chunk in photo_file.chunks():
                    destination.write(chunk)
            
            return JsonResponse({
                'success': True, 
                'message': 'Fotoğraf başarıyla yüklendi',
                'filename': filename
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'Yükleme hatası: {str(e)}'})
    
    return JsonResponse({'success': False, 'message': 'Geçersiz istek'})


def label_all_faces(request):
    """Grouped faces'i film seçerek, grup bazlı etiketle"""
    grouped_dir = os.path.join(settings.MEDIA_ROOT, 'grouped_faces')
    labeled_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces')
    os.makedirs(grouped_dir, exist_ok=True)
    os.makedirs(labeled_dir, exist_ok=True)

    selected_movie = request.GET.get('movie') or request.POST.get('movie')
    
    # Film seçilmemişse, film listesini göster
    if not selected_movie and request.method == 'GET':
        movies_with_groups = []
        if os.path.isdir(grouped_dir):
            for movie_folder in sorted(os.listdir(grouped_dir)):
                movie_path = os.path.join(grouped_dir, movie_folder)
                if os.path.isdir(movie_path):
                    group_count = len([f for f in os.listdir(movie_path) if f.startswith('celebrity_') and os.path.isdir(os.path.join(movie_path, f))])
                    if group_count > 0:
                        movies_with_groups.append({'name': movie_folder, 'group_count': group_count})
        
        return render(request, 'core/label_face_filesystem.html', {
            'action': 'select_movie',
            'movies': movies_with_groups
        })
    
    if not selected_movie:
        return redirect('core:label_all_faces')
    
    movie_path = os.path.join(grouped_dir, selected_movie)
    if not os.path.isdir(movie_path):
        return redirect('core:label_all_faces')
    
    # POST isteği - grup etiketle veya sil
    if request.method == 'POST':
        action = request.POST.get('action')
        group_id = request.POST.get('group_id')
        
        if action == 'discard' and group_id:
            group_path = os.path.join(movie_path, group_id)
            if os.path.isdir(group_path):
                shutil.rmtree(group_path)
        
        elif action == 'save' and group_id:
            cast_name = request.POST.get('assigned_cast_name')
            if cast_name:
                safe_cast_name = get_safe_filename(cast_name)
                new_cast_dir = os.path.join(labeled_dir, safe_cast_name)
                os.makedirs(new_cast_dir, exist_ok=True)
                
                group_path = os.path.join(movie_path, group_id)
                if os.path.isdir(group_path):
                    for filename in os.listdir(group_path):
                        src = os.path.join(group_path, filename)
                        if os.path.isfile(src):
                            dst = os.path.join(new_cast_dir, f"{selected_movie}_{group_id}_{filename}")
                            shutil.move(src, dst)
                    os.rmdir(group_path)
        
        return redirect(f'{reverse("core:label_all_faces")}?movie={selected_movie}')
    
    # GET - grupları göster
    all_groups = []
    for group_folder in sorted(os.listdir(movie_path)):
        if group_folder.startswith('celebrity_'):
            group_path = os.path.join(movie_path, group_folder)
            if os.path.isdir(group_path):
                faces = sorted([f for f in os.listdir(group_path) if f.endswith('.jpg')])
                if faces:
                    all_groups.append({
                        'id': group_folder,
                        'faces': [f'grouped_faces/{selected_movie}/{group_folder}/{f}' for f in faces]
                    })
    
    # Film cast listesi - sadece o filmin oyuncuları
    movie_cast_list = []
    all_actors_list = []
    try:
        # Filmi dosya sistemi adına göre bul (case-sensitive)
        movie = Movie.objects.all().first()
        for m in Movie.objects.all():
            if get_safe_filename(m.title) == selected_movie or m.title == selected_movie:
                movie = m
                break
        
        print(f"[DEBUG] Movie found: {movie}")
        print(f"[DEBUG] Searching for selected_movie: {selected_movie}")
        
        if movie:
            cast_members = MovieCast.objects.filter(movie=movie).select_related('actor').order_by('actor__name')
            movie_cast_list = [cast.actor.name for cast in cast_members]
            print(f"[DEBUG] Movie cast count: {len(movie_cast_list)}")
            print(f"[DEBUG] Cast list: {movie_cast_list}")
        else:
            print(f"[DEBUG] Movie not found in database for: {selected_movie}")
        
        # Tüm aktörleri de al (cast'ta olmayan için)
        from .models import Actor
        all_actors_list = [actor.name for actor in Actor.objects.all().order_by('name')]
        print(f"[DEBUG] All actors count: {len(all_actors_list)}")
    except Exception as e:
        print(f"[ERROR] Cast fetch error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"[DEBUG] Rendering with cast_list={len(movie_cast_list)}, all_actors={len(all_actors_list)}")
    
    return render(request, 'core/label_face_filesystem.html', {
        'action': 'label_groups',
        'movie_title': selected_movie,
        'groups': all_groups,
        'cast_list': movie_cast_list,  # Filmin cast'ı
        'all_actors': all_actors_list  # Tüm aktörler
    })


from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def delete_movie(request):
    """Delete all grouped faces for a movie (AJAX endpoint)."""
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
            movie_name = data.get('movie_name', '')
            
            print(f"[DELETE_MOVIE] Received request for: {movie_name}")
            
            if not movie_name:
                return JsonResponse({'success': False, 'error': 'Movie name required'}, status=400)
            
            # Security: validate movie name (no path traversal)
            if '..' in movie_name or '/' in movie_name or '\\' in movie_name:
                print(f"[DELETE_MOVIE] Invalid movie name: {movie_name}")
                return JsonResponse({'success': False, 'error': 'Invalid movie name'}, status=400)
            
            grouped_faces_dir = os.path.join(settings.MEDIA_ROOT, 'grouped_faces')
            movie_path = os.path.join(grouped_faces_dir, movie_name)
            
            print(f"[DELETE_MOVIE] Movie path: {movie_path}")
            print(f"[DELETE_MOVIE] Exists: {os.path.exists(movie_path)}")
            print(f"[DELETE_MOVIE] Is dir: {os.path.isdir(movie_path)}")
            
            # Security: ensure path is within grouped_faces
            if not movie_path.startswith(grouped_faces_dir):
                print(f"[DELETE_MOVIE] Path security check failed")
                return JsonResponse({'success': False, 'error': 'Invalid path'}, status=403)
            
            if os.path.exists(movie_path) and os.path.isdir(movie_path):
                try:
                    shutil.rmtree(movie_path)
                    print(f"[DELETE_MOVIE] ✓ Successfully deleted: {movie_path}")
                    return JsonResponse({'success': True}, status=200)
                except Exception as e:
                    print(f"[DELETE_MOVIE] ✗ Error deleting: {e}")
                    return JsonResponse({'success': False, 'error': str(e)}, status=500)
            else:
                print(f"[DELETE_MOVIE] ✗ Movie folder not found or not a directory")
                return JsonResponse({'success': False, 'error': 'Movie folder not found'}, status=404)
        except json.JSONDecodeError as e:
            print(f"[DELETE_MOVIE] ✗ JSON parse error: {e}")
            return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            print(f"[DELETE_MOVIE] ✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
    return JsonResponse({'success': False, 'error': 'Invalid request'}, status=405)


def serve_optimized_image(request, image_path):
    """
    ✅ PERFORMANS OPTİMİZASYONU: Resim cache'i ve compression ile serve et
    
    Kullanım: 
    - Template: <img data-src="{% url 'core:serve_optimized_image' image_path=image_path %}?w=150&q=80">
    """
    from PIL import Image
    import io
    
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


def actors_dashboard(request):
    """
    ✅ OPTIMIZED DASHBOARD: Hızlı yükleme ile kalite analizi
    
    ⚡ Optimizasyonlar:
    1. Django cache kullan (5 dakika TTL)
    2. Pagination (15 aktör/sayfa)
    3. Kalite skorlarını lazy-load et (AJAX)
    4. Dosya sistemi taramasını optimize et
    """
    from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
    
    labeled_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces')
    os.makedirs(labeled_dir, exist_ok=True)
    
    search_query = request.GET.get('search', '').strip()
    page = request.GET.get('page', 1)
    
    # Cache key oluştur
    cache_key = f"actors_list_{search_query}"
    
    # Cache'ten kontrol et (5 dakika)
    actors_data = cache.get(cache_key)
    
    if actors_data is None:
        print(f"[CACHE] Cache miss for {cache_key}, calculating...")
        actors_data = []
        
        if os.path.isdir(labeled_dir):
            # ⚡ Optimize: Tek bir os.listdir çağrısı
            actor_folders = sorted([f for f in os.listdir(labeled_dir) 
                                   if os.path.isdir(os.path.join(labeled_dir, f))])
            
            for actor_folder in actor_folders:
                # Arama filtresi hemen uygula (disk taramasını azalt)
                if search_query and search_query.lower() not in actor_folder.lower():
                    continue
                
                actor_path = os.path.join(labeled_dir, actor_folder)
                
                # ⚡ Optimize: os.listdir sadece jpg dosyaları için
                photos = [f for f in os.listdir(actor_path) if f.endswith('.jpg')]
                
                if not photos:
                    continue
                
                # ⚡ Optimize: Filmler set'i sadece 1 loop'ta
                movies_set = set()
                for photo in photos:
                    movie_name = photo.split('_')[0]  # İlk kısım film adı
                    movies_set.add(movie_name)
                
                # ⚡ Optimize: Preview fotoları sadece dosya adları (path'ler değil)
                preview_photos = sorted(photos)[:5]
                
                actors_data.append({
                    'name': actor_folder,
                    'display_name': actor_folder.replace('_', ' ').title(),
                    'photo_count': len(photos),
                    'movie_count': len(movies_set),
                    'movies': sorted(list(movies_set)),
                    'preview_photos': preview_photos,  # Sadece dosya adları
                })
        
        # Cache'e kaydet (5 dakika)
        cache.set(cache_key, actors_data, 300)
        print(f"[CACHE] Cached {len(actors_data)} actors for 5 minutes")
    else:
        print(f"[CACHE] Cache hit for {cache_key}")
    
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
    labeled_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces', actor_name)
    
    if not os.path.isdir(labeled_dir):
        return redirect('core:actors_dashboard')
    
    # Tüm fotoğrafları al
    photos = sorted([f for f in os.listdir(labeled_dir) if f.endswith('.jpg')])
    photo_paths = [f'labeled_faces/{actor_name}/{p}' for p in photos]
    
    # Filmler
    movies_set = set()
    for photo in photos:
        parts = photo.split('_')
        if len(parts) > 0:
            movie_name = parts[0]
            movies_set.add(movie_name)
    
    # Fotoğraf listesi oluştur
    photo_data = []
    for photo in photos:
        photo_data.append({
            'filename': photo,
            'path': f'labeled_faces/{actor_name}/{photo}',
        })
    
    context = {
        'actor_name': actor_name,
        'display_name': actor_name.replace('_', ' ').title(),
        'photos': photo_data,
        'photo_count': len(photos),
        'movie_count': len(movies_set),
        'movies': sorted(list(movies_set)),
    }
    
    return render(request, 'core/actor_detail.html', context)
