import os
import shutil
import time
from django.conf import settings
from core.utils.file_utils import get_safe_filename

def get_grouped_faces_dir():
    return os.path.join(settings.MEDIA_ROOT, 'grouped_faces')

def get_unlabeled_faces_dir():
    return os.path.join(settings.MEDIA_ROOT, 'unlabeled_faces')

def get_labeled_faces_dir():
    return os.path.join(settings.MEDIA_ROOT, 'labeled_faces')

def count_pending_groups():
    grouped_faces_dir = get_grouped_faces_dir()
    pending_groups_count = 0
    if os.path.exists(grouped_faces_dir):
        for movie_folder in os.listdir(grouped_faces_dir):
            movie_path = os.path.join(grouped_faces_dir, movie_folder)
            if os.path.isdir(movie_path):
                pending_groups_count += len([f for f in os.listdir(movie_path) if f.startswith('celebrity_') and os.path.isdir(os.path.join(movie_path, f))])
    return pending_groups_count

def count_pending_faces():
    """
    Counts total individual face images waiting to be labeled in grouped_faces.
    """
    grouped_faces_dir = get_grouped_faces_dir()
    pending_faces_count = 0
    if os.path.exists(grouped_faces_dir):
        for movie_folder in os.listdir(grouped_faces_dir):
            movie_path = os.path.join(grouped_faces_dir, movie_folder)
            if os.path.isdir(movie_path):
                # Iterate groups
                for group_folder in os.listdir(movie_path):
                    if group_folder.startswith('celebrity_'):
                        group_path = os.path.join(movie_path, group_folder)
                        if os.path.isdir(group_path):
                            # Count jpgs
                            pending_faces_count += len([f for f in os.listdir(group_path) if f.endswith('.jpg')])
    return pending_faces_count

def list_unlabeled_faces_service():
    unlabeled_dir = get_unlabeled_faces_dir()
    os.makedirs(unlabeled_dir, exist_ok=True)

    unlabeled_faces = []
    if os.path.exists(unlabeled_dir):
        for movie_folder in sorted(os.listdir(unlabeled_dir)):
            movie_path = os.path.join(unlabeled_dir, movie_folder)
            if os.path.isdir(movie_path):
                for filename in sorted(os.listdir(movie_path)):
                    if filename.endswith('.jpg'):
                        unlabeled_faces.append({
                            'path': os.path.join('unlabeled_faces', movie_folder, filename),
                            'movie_title': movie_folder
                        })
    return unlabeled_faces

def delete_single_face_file(face_path):
    # Forward slash'i sistem path'ine çevir
    face_path_parts = face_path.split('/')
    full_path = os.path.join(settings.MEDIA_ROOT, *face_path_parts)
    
    if os.path.isfile(full_path):
        os.remove(full_path)
        return True
    return False

def get_movies_with_groups():
    grouped_dir = get_grouped_faces_dir()
    movies_with_groups = []
    if os.path.isdir(grouped_dir):
        for movie_folder in sorted(os.listdir(grouped_dir)):
            movie_path = os.path.join(grouped_dir, movie_folder)
            if os.path.isdir(movie_path):
                group_count = len([f for f in os.listdir(movie_path) if f.startswith('celebrity_') and os.path.isdir(os.path.join(movie_path, f))])
                if group_count > 0:
                    movies_with_groups.append({'name': movie_folder, 'group_count': group_count})
    return movies_with_groups

from django.core.cache import cache

def get_groups_for_movie(movie_folder, page=1, per_page=20):
    """
    Paginated and cached retrieval of face groups.
    """
    cache_key = f"groups_list_{movie_folder}"
    # Cache duration: 5 minutes (invalidated on delete/save actions)
    all_group_names = cache.get(cache_key)

    movie_path = os.path.join(get_grouped_faces_dir(), movie_folder)
    
    if all_group_names is None:
        if os.path.isdir(movie_path):
            # Only list directories efficiently
            all_group_names = sorted([
                d for d in os.listdir(movie_path) 
                if d.startswith('celebrity_') and os.path.isdir(os.path.join(movie_path, d))
            ])
            cache.set(cache_key, all_group_names, 300)
        else:
            all_group_names = []
            
    # Pagination Logic
    total_groups = len(all_group_names)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    current_page_groups = all_group_names[start_idx:end_idx]
    
    # Detailed scan ONLY for the current page
    groups_data = []
    for group_folder in current_page_groups:
        group_path = os.path.join(movie_path, group_folder)
        # Fast scan of images
        faces = sorted([f for f in os.listdir(group_path) if f.endswith('.jpg')])
        if faces:
            groups_data.append({
                'id': group_folder,
                'faces': [f'grouped_faces/{movie_folder}/{group_folder}/{f}' for f in faces]
            })
            
    return {
        'groups': groups_data,
        'total_count': total_groups,
        'total_pages': (total_groups + per_page - 1) // per_page,
        'current_page': page,
        'has_next': end_idx < total_groups,
        'has_previous': start_idx > 0,
    }

def invalidate_movie_cache(movie_folder):
    cache.delete(f"groups_list_{movie_folder}")


def discard_group(movie_folder, group_id):
    group_path = os.path.join(get_grouped_faces_dir(), movie_folder, group_id)
    if os.path.isdir(group_path):
        shutil.rmtree(group_path)
        invalidate_movie_cache(movie_folder)
        return True
    return False

def save_group_as_actor(movie_folder, group_id, cast_name):
    safe_cast_name = get_safe_filename(cast_name)
    new_cast_dir = os.path.join(get_labeled_faces_dir(), safe_cast_name)
    os.makedirs(new_cast_dir, exist_ok=True)
    
    group_path = os.path.join(get_grouped_faces_dir(), movie_folder, group_id)
    if os.path.isdir(group_path):
        for filename in os.listdir(group_path):
            src = os.path.join(group_path, filename)
            if os.path.isfile(src):
                dst = os.path.join(new_cast_dir, f"{movie_folder}_{group_id}_{filename}")
                shutil.move(src, dst)
        os.rmdir(group_path)
        invalidate_movie_cache(movie_folder)
        return True
    return False

def delete_movie_groups_service(movie_name):
    grouped_faces_dir = get_grouped_faces_dir()
    movie_path = os.path.join(grouped_faces_dir, movie_name)
    
    # Security: ensure path is within grouped_faces
    if not movie_path.startswith(grouped_faces_dir):
         raise ValueError("Invalid path Security Check Failed")

    if os.path.exists(movie_path) and os.path.isdir(movie_path):
        shutil.rmtree(movie_path)
        return True
    return False

def save_actor_photo(actor_name, photo_file):
    actor_dir = os.path.join(get_labeled_faces_dir(), actor_name)
    os.makedirs(actor_dir, exist_ok=True)
    
    ext = os.path.splitext(photo_file.name)[1]
    filename = f"{int(time.time() * 1000)}{ext}"
    file_path = os.path.join(actor_dir, filename)
    

    with open(file_path, 'wb+') as destination:
        for chunk in photo_file.chunks():
            destination.write(chunk)
            
    return filename

def sync_face_groups(movie_name):
    """
    Scans the filesystem for a movie's groups and ensures FaceGroup entries exist.
    OPTIMIZED: Incremental sync. Only reads distinct folders not in DB.
    """
    from core.models import Movie, FaceGroup
    
    movie_path = os.path.join(get_grouped_faces_dir(), movie_name)
    if not os.path.exists(movie_path):
        return
        
    try:
        movie_obj = Movie.objects.filter(title=movie_name).first()
        if not movie_obj:
            # Try looser match
            for m in Movie.objects.all():
                if get_safe_filename(m.title) == movie_name:
                    movie_obj = m
                    break
            if not movie_obj:
                movie_obj = Movie.objects.create(title=movie_name)
    except Exception as e:
        print(f"Sync Init Error: {e}")
        return

    # Scan folders
    try:
        # FS Source of True
        fs_group_ids = set([d for d in os.listdir(movie_path) if d.startswith('celebrity_') and os.path.isdir(os.path.join(movie_path, d))])
        
        # DB State
        db_groups = FaceGroup.objects.filter(movie=movie_obj)
        db_group_ids = set(db_groups.values_list('group_id', flat=True))
        
        # 1. Identify New Groups (FS - DB)
        new_group_ids = fs_group_ids - db_group_ids
        
        if new_group_ids:
            # print(f"Syncing {len(new_group_ids)} new groups for {movie_name}...")
            new_objs = []
            for group_id in new_group_ids:
                group_full_path = os.path.join(movie_path, group_id)
                # Count faces only for new groups
                face_count = len([f for f in os.listdir(group_full_path) if f.endswith('.jpg')])
                
                new_objs.append(FaceGroup(
                    movie=movie_obj,
                    group_id=group_id,
                    face_count=face_count,
                    risk_level='HIGH' # Default safe
                ))
            
            if new_objs:
                FaceGroup.objects.bulk_create(new_objs)

        # 2. Identify Deleted Groups (DB - FS) -> Cleanup DB
        deleted_group_ids = db_group_ids - fs_group_ids
        if deleted_group_ids:
            # print(f"Removing {len(deleted_group_ids)} stale groups from DB...")
            FaceGroup.objects.filter(movie=movie_obj, group_id__in=deleted_group_ids).delete()
            
    except Exception as e:
        print(f"Sync Loop Error: {e}")

