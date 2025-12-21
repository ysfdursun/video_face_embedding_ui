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
                
                # Check for Profile Photo
                profile_photo_url = None
                profile_photo_dir = os.path.join(settings.MEDIA_ROOT, 'Selected_Profiles', actor_folder)
                
                if os.path.isdir(profile_photo_dir):
                    # Get first image file in the directory
                    profile_images = [f for f in os.listdir(profile_photo_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if profile_images:
                        profile_photo_url = f"Selected_Profiles/{actor_folder}/{profile_images[0]}"

                actors_data.append({
                    'name': actor_folder,
                    'display_name': actor_folder.replace('_', ' ').title(),
                    'photo_count': len(photos),
                    'movie_count': len(movies_set),
                    'movies': sorted(list(movies_set)),
                    'preview_photos': preview_photos,
                    'profile_photo_url': profile_photo_url,
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
    
    movie_count = len(movies_set)

    # Profile Photo Logic
    profile_photo_dir = os.path.join(settings.MEDIA_ROOT, 'Selected_Profiles', actor_name)
    profile_photo_url = None
    
    if os.path.isdir(profile_photo_dir):
        profile_images = [f for f in os.listdir(profile_photo_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if profile_images:
            profile_photo_url = f"Selected_Profiles/{actor_name}/{profile_images[0]}"

    return {
        'actor_name': actor_name,
        'display_name': actor_name.replace('_', ' ').title(),
        'photos': photo_data,
        'photo_count': len(photos),
        'movie_count': movie_count,
        'movies': sorted(list(movies_set)),
        'profile_photo_url': profile_photo_url,
    }

def create_actor(name):
    """
    Creates a new actor in the database and filesystem.
    """
    from core.models import Actor
    from core.utils.file_utils import get_safe_filename
    
    safe_name = get_safe_filename(name)
    if not safe_name:
        return None, "Invalid name"
        
    labeled_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces', safe_name)
    
    if Actor.objects.filter(name=safe_name).exists() or os.path.exists(labeled_dir):
        return None, "Actor already exists"
        
    try:
        Actor.objects.create(name=safe_name)
        os.makedirs(labeled_dir, exist_ok=True)
        # Invalidate cache
        cache.delete_pattern("actors_list_*")
        return safe_name, None
    except Exception as e:
        return None, str(e)

def save_profile_photo(actor_name, photo_file):
    """
    Saves or updates the profile photo for an actor in Selected_Profiles.
    Creates a folder for the actor if it doesn't exist.
    """
    profile_photo_dir = os.path.join(settings.MEDIA_ROOT, 'Selected_Profiles', actor_name)
    
    # Create directory if not exists
    os.makedirs(profile_photo_dir, exist_ok=True)
    
    # Remove all existing files in the directory to ensure only one profile photo
    for f in os.listdir(profile_photo_dir):
        os.remove(os.path.join(profile_photo_dir, f))
    
    # Save new file
    # Use actor_name + extension or original name
    ext = os.path.splitext(photo_file.name)[1]
    filename = f"{actor_name}{ext}" 
    file_path = os.path.join(profile_photo_dir, filename)
    
    with open(file_path, 'wb+') as destination:
        for chunk in photo_file.chunks():
            destination.write(chunk)
            
    return filename

def delete_profile_photo(actor_name):
    """
    Deletes the profile photo directory for an actor.
    """
    profile_photo_dir = os.path.join(settings.MEDIA_ROOT, 'Selected_Profiles', actor_name)
    
    if os.path.isdir(profile_photo_dir):
        import shutil
        shutil.rmtree(profile_photo_dir)
        return True
    return False

def delete_actor(name):
    """
    Deletes an actor from the database and filesystem.
    """
    from core.models import Actor
    import shutil
    
    try:
        # DB Delete
        Actor.objects.filter(name=name).delete()
        
        # Filesystem Delete
        labeled_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces', name)
        if os.path.isdir(labeled_dir):
            shutil.rmtree(labeled_dir)
            
        # Invalidate cache
        cache.delete_pattern("actors_list_*")
        return True, None
    except Exception as e:
        return False, str(e)
