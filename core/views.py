from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.urls import reverse
import os
import shutil
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
    
    # Film cast listesi
    movie_cast_list = []
    try:
        movie = Movie.objects.filter(title__iexact=selected_movie).first()
        if movie:
            cast_members = MovieCast.objects.filter(movie=movie).select_related('actor')
            movie_cast_list = sorted([cast.actor.name for cast in cast_members])
    except Exception as e:
        print(f"Cast fetch error: {e}")
    
    dir_cast_list = [d.replace('_', ' ') for d in os.listdir(labeled_dir) if os.path.isdir(os.path.join(labeled_dir, d))]
    combined_cast_list = sorted(list(set(movie_cast_list + dir_cast_list)))
    
    return render(request, 'core/label_face_filesystem.html', {
        'action': 'label_groups',
        'movie_title': selected_movie,
        'groups': all_groups,
        'cast_list': combined_cast_list
    })

