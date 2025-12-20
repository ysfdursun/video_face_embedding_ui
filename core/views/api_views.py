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
            
            movie = Movie.objects.get(title=movie_title)
            actor = Actor.objects.get(name=actor_name)
            
            MovieCast.objects.get_or_create(movie=movie, actor=actor)
            
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)

class ActorSearchAPI(View):
    def get(self, request):
        query = request.GET.get('q', '')
        actors = Actor.objects.filter(name__icontains=query).values('name')[:20]
        return JsonResponse({'results': list(actors)})
