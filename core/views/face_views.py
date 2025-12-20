import json
from django.shortcuts import render, redirect, reverse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from core.models import Movie, MovieCast, Actor
from core.services import face_service
from core.utils.file_utils import get_safe_filename

def list_unlabeled_faces(request):
    """List all unlabeled faces as a gallery."""
    unlabeled_faces = face_service.list_unlabeled_faces_service()
    
    context = {
        'unlabeled_faces': unlabeled_faces,
        'unlabeled_faces_count': len(unlabeled_faces),
    }
    return render(request, 'core/list_unlabeled.html', context)

def delete_single_face(request):
    """AJAX ile tek bir fotoğrafı sil"""
    if request.method == 'POST':
        face_path = request.POST.get('face_path')
        if face_path:
            # Service call
            if face_service.delete_single_face_file(face_path):
                return JsonResponse({'success': True})
            else:
                return JsonResponse({'success': False, 'message': 'Dosya bulunamadı'})
        return JsonResponse({'success': False, 'message': 'face_path eksik'})
    return JsonResponse({'success': False, 'message': 'Geçersiz istek'})

def label_all_faces(request):
    """Grouped faces'i film seçerek, grup bazlı etiketle"""
    
    selected_movie = request.GET.get('movie') or request.POST.get('movie')
    
    # Film seçilmemişse, film listesini göster
    if not selected_movie and request.method == 'GET':
        movies_with_groups = face_service.get_movies_with_groups()
        return render(request, 'core/label_face_filesystem.html', {
            'action': 'select_movie',
            'movies': movies_with_groups
        })
    
    if not selected_movie:
        return redirect('core:label_all_faces')
    
    # AJAX Handler for Discard/Save
    if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        data = json.loads(request.body)
        action = data.get('action')
        group_id = data.get('group_id')
        movie_from_post = data.get('movie')  # Title from frontend
        
        if not group_id:
             return JsonResponse({'success': False, 'message': 'Missing group_id'})

        if action == 'discard':
            if face_service.discard_group(selected_movie, group_id):
                return JsonResponse({'success': True})
            return JsonResponse({'success': False, 'message': 'Failed to discard'})
        
        elif action == 'save':
            cast_name = data.get('assigned_cast_name')
            if cast_name:
                # 1. DB LINKING
                try:
                    actor, _ = Actor.objects.get_or_create(name=cast_name)
                    
                    target_movie = None
                    if movie_from_post:
                        target_movie = Movie.objects.filter(title=movie_from_post).first()
                    
                    if not target_movie:
                        # Fallback to safe slug matching
                        for m in Movie.objects.all():
                            if get_safe_filename(m.title) == selected_movie or m.title == selected_movie:
                                target_movie = m
                                break
                    
                    if target_movie:
                        MovieCast.objects.get_or_create(movie=target_movie, actor=actor)
                except Exception as e:
                    print(f"DB Link Error: {e}")

                # 2. FILE MOVING
                if face_service.save_group_as_actor(selected_movie, group_id, cast_name):
                    return JsonResponse({'success': True})
                return JsonResponse({'success': False, 'message': 'Failed to save'})
            return JsonResponse({'success': False, 'message': 'Missing cast name'})
            
        return JsonResponse({'success': False, 'message': 'Invalid action'})

    # Legacy POST handler (fallback)
    if request.method == 'POST':
        action = request.POST.get('action')
        group_id = request.POST.get('group_id')
        
        if action == 'discard' and group_id:
            face_service.discard_group(selected_movie, group_id)
        
        elif action == 'save' and group_id:
            cast_name = request.POST.get('assigned_cast_name')
            if cast_name:
                face_service.save_group_as_actor(selected_movie, group_id, cast_name)
        
        return redirect(f'{reverse("core:label_all_faces")}?movie={selected_movie}')
    
    # GET - Pagination
    page = int(request.GET.get('page', 1))
    
    # Fetch paginated data from service
    result = face_service.get_groups_for_movie(selected_movie, page=page, per_page=20)
    
    # Film cast listesi
    movie_cast_list = []
    all_actors_list = []
    try:
        movie = Movie.objects.all().first()
        for m in Movie.objects.all():
            if get_safe_filename(m.title) == selected_movie or m.title == selected_movie:
                movie = m
                break
        
        if movie:
            cast_members = MovieCast.objects.filter(movie=movie).select_related('actor').order_by('actor__name')
            movie_cast_list = [cast.actor.name for cast in cast_members]
        
        all_actors_list = [actor.name for actor in Actor.objects.all().order_by('name')]
    except Exception as e:
        print(f"[ERROR] Cast fetch error: {e}")
    
    return render(request, 'core/label_face_filesystem.html', {
        'action': 'label_groups',
        'movie_title': selected_movie,
        'groups': result['groups'],
        'pagination': result,  # Pass pagination metadata
        'cast_list': movie_cast_list,
        'all_actors': all_actors_list
    })

@csrf_exempt
def delete_movie(request):
    """Delete all grouped faces for a movie (AJAX endpoint)."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            movie_name = data.get('movie_name', '')
            
            if not movie_name:
                return JsonResponse({'success': False, 'error': 'Movie name required'}, status=400)
            
            # Security: validate movie name (no path traversal logic in service)
            
            try:
                if face_service.delete_movie_groups_service(movie_name):
                    return JsonResponse({'success': True}, status=200)
                else:
                    return JsonResponse({'success': False, 'error': 'Movie folder not found'}, status=404)
            except ValueError as ve:
                return JsonResponse({'success': False, 'error': str(ve)}, status=403)
            except Exception as e:
                 return JsonResponse({'success': False, 'error': str(e)}, status=500)

        except json.JSONDecodeError as e:
            return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
    return JsonResponse({'success': False, 'error': 'Invalid request'}, status=405)
