import os
from django.conf import settings
from django.core.cache import cache

def get_actors_data(search_query=''):
    """
    Retrieves actor data from the filesystem or cache.
    Returns a list of actor dictionaries.
    """
    labeled_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces')
    
    # Cache key
    cache_key = f"actors_list_{search_query}"
    actors_data = cache.get(cache_key)
    
    if actors_data is None:
        actors_data = []
        if os.path.isdir(labeled_dir):
            # Optimize: Single os.listdir
            actor_folders = sorted([f for f in os.listdir(labeled_dir) 
                                if os.path.isdir(os.path.join(labeled_dir, f))])
            
            for actor_folder in actor_folders:
                # Filter by search query
                if search_query and search_query.lower() not in actor_folder.lower():
                    continue
                
                actor_path = os.path.join(labeled_dir, actor_folder)
                
                # Optimize: os.listdir only for jpg
                photos = [f for f in os.listdir(actor_path) if f.endswith('.jpg')]
                
                if not photos:
                    continue
                
                # Optimize: Movies set in one loop
                movies_set = set()
                for photo in photos:
                    movie_name = photo.split('_')[0]  # First part is movie name
                    movies_set.add(movie_name)
                
                # Optimize: Preview photos
                preview_photos = sorted(photos)[:5]
                
                actors_data.append({
                    'name': actor_folder,
                    'display_name': actor_folder.replace('_', ' ').title(),
                    'photo_count': len(photos),
                    'movie_count': len(movies_set),
                    'movies': sorted(list(movies_set)),
                    'preview_photos': preview_photos,
                })
        
        # Cache for 5 minutes
        cache.set(cache_key, actors_data, 300)
    
    return actors_data

def get_actor_details(actor_name):
    """
    Retrieves details for a specific actor.
    """
    labeled_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces', actor_name)
    
    if not os.path.isdir(labeled_dir):
        return None
    
    photos = sorted([f for f in os.listdir(labeled_dir) if f.endswith('.jpg')])
    
    movies_set = set()
    photo_data = []
    
    for photo in photos:
        # Movies set
        parts = photo.split('_')
        if len(parts) > 0:
            movie_name = parts[0]
            movies_set.add(movie_name)
            
        # Photo data
        photo_data.append({
            'filename': photo,
            'path': f'labeled_faces/{actor_name}/{photo}',
        })
    
    return {
        'actor_name': actor_name,
        'display_name': actor_name.replace('_', ' ').title(),
        'photos': photo_data,
        'photo_count': len(photos),
        'movie_count': len(movies_set),
        'movies': sorted(list(movies_set)),
    }
