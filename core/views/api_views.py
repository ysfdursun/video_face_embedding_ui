from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
from core.models import Movie, Actor, MovieCast
from core.services import actor_service, video_service

@method_decorator(csrf_exempt, name='dispatch')
class MovieCreateAPI(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            title = data.get('title')
            if not title:
                return JsonResponse({'success': False, 'message': 'Title is required'}, status=400)
            
            movie, error = video_service.create_movie(title)
            if error:
                return JsonResponse({'success': False, 'message': error}, status=400)
            
            return JsonResponse({
                'success': True, 
                'movie': {'id': movie.id, 'title': movie.title}
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)

@method_decorator(csrf_exempt, name='dispatch')
class ActorCreateAPI(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            name = data.get('name')
            if not name:
                return JsonResponse({'success': False, 'message': 'Name is required'}, status=400)
            
            safe_name, error = actor_service.create_actor(name)
            if error:
                return JsonResponse({'success': False, 'message': error}, status=400)
            
            return JsonResponse({
                'success': True,
                'actor': {'name': safe_name}
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)

@method_decorator(csrf_exempt, name='dispatch')
class ActorDeleteAPI(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            name = data.get('name')
            if not name:
                return JsonResponse({'success': False, 'message': 'Name is required'}, status=400)
            
            success, error = actor_service.delete_actor(name)
            if not success:
               return JsonResponse({'success': False, 'message': error}, status=400)
            
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)

@method_decorator(csrf_exempt, name='dispatch')
class CastManageAPI(View):
    def get(self, request):
        """Get cast for a movie"""
        movie_title = request.GET.get('movie')
        if not movie_title:
             return JsonResponse({'success': False, 'message': 'Movie required'}, status=400)
        
        try:
            movie = Movie.objects.get(title=movie_title)
            cast = MovieCast.objects.filter(movie=movie).select_related('actor')
            return JsonResponse({
                'success': True,
                'cast': [{'name': c.actor.name} for c in cast]
            })
        except Movie.DoesNotExist:
             return JsonResponse({'success': False, 'message': 'Movie not found'}, status=404)

    def post(self, request):
        """Add actor to cast"""
        try:
            data = json.loads(request.body)
            movie_title = data.get('movie')
            actor_name = data.get('actor')
            
            if not movie_title or not actor_name:
                return JsonResponse({'success': False, 'message': 'Movie and Actor required'}, status=400)
            
            # Robust Movie Lookup
            from core.utils.file_utils import get_safe_filename
            movie = Movie.objects.filter(title=movie_title).first()
            
            if not movie:
                 # Fallback: check safe filenames
                 for m in Movie.objects.all():
                    if get_safe_filename(m.title) == movie_title or m.title == movie_title:
                        movie = m
                        break
            
            if not movie:
                return JsonResponse({'success': False, 'message': f'Movie "{movie_title}" not found'}, status=404)

            try:
                actor = Actor.objects.get(name=actor_name)
            except Actor.DoesNotExist:
                return JsonResponse({'success': False, 'message': f'Actor "{actor_name}" not found in DB'}, status=404)
            
            MovieCast.objects.get_or_create(movie=movie, actor=actor)
            
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)

    def delete(self, request):
        """Remove actor from cast"""
        try:
            data = json.loads(request.body)
            movie_title = data.get('movie')
            actor_name = data.get('actor')
            
            if not movie_title or not actor_name:
                return JsonResponse({'success': False, 'message': 'Movie and Actor required'}, status=400)
            
            # Robust Movie Lookup
            from core.utils.file_utils import get_safe_filename
            movie = Movie.objects.filter(title=movie_title).first()
            if not movie:
                 for m in Movie.objects.all():
                    if get_safe_filename(m.title) == movie_title or m.title == movie_title:
                        movie = m
                        break
            
            if not movie:
                return JsonResponse({'success': False, 'message': 'Movie not found'}, status=404)

            actor = Actor.objects.filter(name=actor_name).first()
            if not actor:
                return JsonResponse({'success': False, 'message': 'Actor not found'}, status=404)
            
            MovieCast.objects.filter(movie=movie, actor=actor).delete()
            
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)

class ActorSearchAPI(View):
    def get(self, request):
        query = request.GET.get('q', '').strip()
        if not query:
            return JsonResponse({'results': []})
        
        actors = Actor.objects.filter(name__icontains=query)[:20]
        return JsonResponse({
            'results': [actor.name for actor in actors]
        })

@method_decorator(csrf_exempt, name='dispatch')
class ProcessingStatusAPI(View):
    def get(self, request):
        movie_name = request.GET.get('movie')
        if not movie_name:
            return JsonResponse({'status': 'unknown', 'message': 'Movie name missing'})
            
        from django.core.cache import cache
        # movie_name from GET is "Fight_Club" (safe from template)
        # We need to construct the key exactly like in face_processor
        # In processor: f"processing_status_{get_safe_filename(movie_title)}"
        # Here we assume movie_name is already safe because it comes from frontend {{ movie_title }} 
        # which comes from os.path.splitext... 
        # But let's apply get_safe_filename just to be 100% sure we match the backend logic
        from core.utils.file_utils import get_safe_filename
        safe_name = get_safe_filename(movie_name)
        
        cache_key = f"processing_status_{safe_name}" 
        status = cache.get(cache_key)
        
        # DEBUG LOG
        if status != 'running': # Don't spam logs if validly running
             print(f"[StatusAPI] Checking for '{movie_name}' -> Key: '{cache_key}' -> Status: {status}")
        
        return JsonResponse({'status': status or 'unknown'})
