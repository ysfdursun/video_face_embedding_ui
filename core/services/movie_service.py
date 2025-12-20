from django.core.paginator import Paginator
from django.db.models import Count, Prefetch
from core.models import Movie, MovieCast, Actor
from django.conf import settings
import os

def get_movies_list(page=1, per_page=12, search_query=''):
    """
    Returns a paginated list of movies with actor counts and preview faces.
    """
    queryset = Movie.objects.all().annotate(actor_count=Count('cast_members'))
    
    if search_query:
        queryset = queryset.filter(title__icontains=search_query)
        
    paginator = Paginator(queryset, per_page)
    movies_page = paginator.get_page(page)
    
    # Enrich with preview faces (Post-processing to avoid complex subqueries)
    # We fetch the first 4 actors for each movie on the current page efficiently
    movie_ids = [m.id for m in movies_page]
    
    # Fetch cast members for these movies
    cast_members = MovieCast.objects.filter(movie_id__in=movie_ids).select_related('actor')
    
    # Group by movie
    movie_cast_map = {}
    for cast in cast_members:
        if cast.movie_id not in movie_cast_map:
            movie_cast_map[cast.movie_id] = []
        
        # Limit to 4 for preview
        if len(movie_cast_map[cast.movie_id]) < 4:
            # Try to find a face image for the actor
            actor_name = cast.actor.name
            face_path = _get_first_face_for_actor(actor_name)
            if face_path:
                movie_cast_map[cast.movie_id].append({
                    'name': actor_name,
                    'image': face_path
                })
    
    # Attach to movie objects
    for movie in movies_page:
        movie.preview_cast = movie_cast_map.get(movie.id, [])
        
    return movies_page

def get_movie_details(movie_id):
    """
    Returns movie details and full cast.
    """
    try:
        movie = Movie.objects.prefetch_related('cast_members__actor').get(id=movie_id)
        
        # Enrich cast with photos
        cast_list = []
        for cm in movie.cast_members.all():
            actor = cm.actor
            face_path = _get_first_face_for_actor(actor.name)
            
            cast_list.append({
                'name': actor.name,
                'display_name': actor.name.replace('_', ' ').title(),
                'image': face_path,
                # We could add more stats here if needed
            })
            
        return {
            'movie': movie,
            'cast': cast_list,
            'actor_count': len(cast_list),
            'cast_names': [c['name'] for c in cast_list],
            'all_actors': [a.name for a in Actor.objects.all().order_by('name')]
        }
    except Movie.DoesNotExist:
        return None

def _get_first_face_for_actor(actor_name):
    """
    Helper to find the first jpg face for an actor.
    Returns relative path or None.
    """
    # We can use cache here if needed, but file system check is fast enough for top 4
    # Optimization: Maybe cache this path in the Actor model or separate cache key?
    # For now, let's just check FS.
    
    actor_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces', actor_name)
    if os.path.isdir(actor_dir):
        # Just grab the first one
        try:
            files = sorted(os.listdir(actor_dir))
            for f in files:
                if f.lower().endswith('.jpg'):
                    return f'labeled_faces/{actor_name}/{f}'
        except OSError:
            pass
    return None
