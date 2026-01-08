import os
import shutil
import time
import pickle
import numpy as np
from django.conf import settings
from core.models import Actor, Movie, MovieCast
from core.utils.file_utils import get_safe_filename
from core.face_recognizer import VideoFaceRecognizer

# Re-use config
from core.config import Config

PKL_PATH = os.path.join(settings.BASE_DIR, 'embeddings_all.pkl')

def enroll_guest_as_actor(session_id, guest_name, target_actor_name, movie_id=None):
    """
    Promote a session guest to a permanent actor.
    1. Move image to labeled_faces
    2. Register Actor in DB
    3. Extract Embedding & Update PKL
    """
    
    # 1. Locate Guest Image
    guest_dir = os.path.join(settings.MEDIA_ROOT, 'temp_faces', session_id)
    guest_image_path = os.path.join(guest_dir, f"{guest_name}.jpg")
    
    if not os.path.exists(guest_image_path):
        return {'success': False, 'error': 'Guest image not found'}
        
    # 2. Prepare Target Directory
    safe_actor_name = get_safe_filename(target_actor_name)
    target_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces', safe_actor_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # 3. Create Unique Filename
    # Format: {timestamp}_{guest_id}.jpg
    timestamp = int(time.time() * 1000)
    new_filename = f"{timestamp}_{guest_name}.jpg"
    target_path = os.path.join(target_dir, new_filename)
    
    # 4. Move File (Copy first to be safe, then maybe we keep temp for session consistency? 
    # Actually, let's copy so session doesn't break if user continues watching)
    try:
        shutil.copy2(guest_image_path, target_path)
    except Exception as e:
        return {'success': False, 'error': f"File move failed: {e}"}
        
    # 5. DB Registration
    try:
        actor, created = Actor.objects.get_or_create(name=target_actor_name)
        
        # Link to movie if provided
        if movie_id:
            try:
                movie = Movie.objects.get(id=movie_id)
                MovieCast.objects.get_or_create(movie=movie, actor=actor)
            except Movie.DoesNotExist:
                pass
                
    except Exception as e:
        # DB error shouldn't stop file/pkl logic but it's bad
        print(f"DB Error: {e}")

    # 6. PKL Update (Embedding)
    try:
        # Check for saved embedding (.npy)
        guest_npy_path = os.path.join(guest_dir, f"{guest_name}.npy")
        
        embedding = None
        
        if os.path.exists(guest_npy_path):
            try:
                embedding = np.load(guest_npy_path)
                print(f"✅ Loaded saved embedding for {guest_name}")
            except Exception as e:
                print(f"⚠️ Failed to load NPY: {e}")
        
        if embedding is None:
            # Fallback: Re-calculate
            print("⚠️ NPY not found. Attempting re-calculation from image...")
            
            # Initialize recognizer just for extraction tools
            from core.model_loader import load_recognizer, load_detector
            
            detector = load_detector()
            recognizer_model = load_recognizer()
            
            if not detector or not recognizer_model:
                 return {'success': False, 'error': "Models not loaded"}
                 
            import cv2
            img = cv2.imread(target_path)
            if img is None:
                 return {'success': False, 'error': "Image read failed"}
                 
            # Detect Face
            # Since the image is a crop, we might need relaxed detection
            # or treat the whole image as a face if we can align it directly.
            # But ArcFace needs landmarks.
            
            faces = detector.get(img)
            if not faces:
                 return {'success': False, 'error': "Face detection failed on saved image (try re-analyzing video)"}
            
            # Take the largest face
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            face = faces[0]
            
            # Align
            lmk = face.kps
            M = cv2.estimateAffinePartial2D(lmk, Config.REF_LANDMARKS, method=cv2.RANSAC)[0]
            if M is None:
                return {'success': False, 'error': "Alignment failed"}
                
            aligned_face = cv2.warpAffine(img, M, Config.ALIGNED_FACE_SIZE)
            
            # Extract
            embedding = recognizer_model.get_feat(aligned_face).flatten()
            embedding = embedding / np.linalg.norm(embedding)
        
        # Update PKL
        _update_pkl_atomic(target_actor_name, embedding)
        
        return {
            'success': True, 
            'actor_name': target_actor_name, 
            'image_url': f"{settings.MEDIA_URL}labeled_faces/{safe_actor_name}/{new_filename}"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': f"Embedding/PKL Error: {e}"}

def _update_pkl_atomic(name, new_embedding, filename=None):
    """
    Load PKL, append embedding and filename, save atomically.
    """
    if not os.path.exists(PKL_PATH):
        data = {}
    else:
        try:
            with open(PKL_PATH, 'rb') as f:
                data = pickle.load(f)
        except:
            data = {}
            
    # Update logic
    if name not in data:
        data[name] = {'templates': [new_embedding], 'files': [filename] if filename else []}
    else:
        entry = data[name]
        # Normalize entry
        if isinstance(entry, dict):
            if 'templates' not in entry: entry['templates'] = []
            if 'files' not in entry: entry['files'] = []
            
            # Check for duplicates?
            # Simple dot product check
            is_dup = False
            for existing in entry['templates']:
                sim = np.dot(new_embedding, existing)
                if sim > 0.99: # Practically identical
                    is_dup = True
                    break
            
            if not is_dup:
                entry['templates'].append(new_embedding)
                if filename:
                    # Only append filename if we appended template, OR append anyway?
                    # Generally templates and files should be synced.
                    entry['files'].append(filename)
            elif filename and filename not in entry['files']:
                # If template is duplicate but filename is new (e.g. same face different photo),
                # we SHOULD append the file but NOT the template?
                # NO. Our system relies on sync. If we don't add template, we shouldn't add file.
                # BUT, wait. If we don't add file, UI thinks it's not embedded.
                # Actually, `toggle_embedding` adds BOTH.
                # If `is_dup` is True, it means we already have this face.
                # If we don't add it to `files`, `get_actor_details` will say "Not Embedded".
                # So we MUST add filename to `files` if it's not there, even if template is duplicate.
                # BUT `toggle_embedding` assumes len(files) == len(templates).
                # Current system is fragile about this sync. 
                # Ideally, we should ADD the template even if duplicate if it's a new file?
                # Or just add filename. 
                # Let's stick to: Add both. The cost of one duplicate vector is synonymous with correct mapping.
                # Overriding duplication check: Add ALWAYS.
                # (The duplication check above was probably premature optimization).
                entry['templates'].append(new_embedding)
                if filename:
                    entry['files'].append(filename)

        else:
            # Legacy format (list or array)
            # Convert to dict
            old_vecs = []
            if isinstance(entry, list): old_vecs = entry
            elif isinstance(entry, np.ndarray): old_vecs = [entry]
            
            data[name] = {'templates': old_vecs, 'files': [filename] if filename else []}
            data[name]['templates'].append(new_embedding)
            
    # Save
    temp_path = PKL_PATH + '.tmp'
    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(data, f)
        shutil.move(temp_path, PKL_PATH)
        
        # Trigger Hot-Reload Signal
        from django.core.cache import cache
        cache.set("recognition_db_version", time.time())
        
        print(f"✅ Updated PKL for {name} (File: {filename})")
    except Exception as e:
        print(f"❌ PKL Save Failed: {e}")

def enroll_uploaded_image(actor_name, image_file):
    """
    Enrolls a directly uploaded image (Django File object).
    1. Saves file to labeled_faces
    2. Updates Actor DB
    3. Extracts Embedding & Updates PKL
    """
    from core.services.face_service import save_actor_photo
    from core.model_loader import load_recognizer, load_detector
    
    # 1. Save File
    filename = save_actor_photo(actor_name, image_file)
    if not filename:
        return {'success': False, 'error': 'Failed to save file'}
        
    safe_actor_name = get_safe_filename(actor_name)
    file_path = os.path.join(settings.MEDIA_ROOT, 'labeled_faces', safe_actor_name, filename)
    
    # 2. DB Update
    try:
        Actor.objects.get_or_create(name=actor_name)
    except Exception as e:
        print(f"DB Error: {e}")

    # 3. Extract & Update PKL
    try:
        # Load models
        detector = load_detector()
        recognizer_model = load_recognizer()
        
        if not detector or not recognizer_model:
             return {'success': True, 'warning': "Models stuck, saved file but PKL not updated."}
             
        import cv2
        img = cv2.imread(file_path)
        if img is None:
             return {'success': True, 'warning': "Could not read saved file for PKL update."}
             
        # Detect
        faces = detector.get(img)
        if not faces:
             # Try stricter alignment if detection fails? Or just skip PKL
             return {'success': True, 'warning': "Face not valid for embedding (but saved)."}
        
        # Largest Face
        faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        face = faces[0]
        
        # Align
        lmk = face.kps
        M = cv2.estimateAffinePartial2D(lmk, Config.REF_LANDMARKS, method=cv2.RANSAC)[0]
        if M is None:
             return {'success': True, 'warning': "Alignment failed for PKL."}
        
        aligned_face = cv2.warpAffine(img, M, Config.ALIGNED_FACE_SIZE)
        
        # Extract
        embedding = recognizer_model.get_feat(aligned_face).flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        # Update PKL
        _update_pkl_atomic(actor_name, embedding, filename)
        
        return {'success': True, 'filename': filename}
        
    except Exception as e:
        print(f"Enroll Error: {e}")
        return {'success': True, 'warning': f"Saved but PKL update failed: {e}"}
