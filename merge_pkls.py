import pickle
import numpy as np
import os
import shutil
from datetime import datetime

TARGET_FILE = 'embeddings_all.pkl'
SOURCE_FILE = 'embeddings_all_.pkl'

def load_pkl(path):
    if not os.path.exists(path):
        print(f"❌ Not found: {path}")
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(data, path):
    # Atomic write
    temp = path + '.tmp'
    with open(temp, 'wb') as f:
        pickle.dump(data, f)
    shutil.move(temp, path)

def get_vectors(entry):
    # Normalize structure to list of arrays
    if isinstance(entry, dict):
        if 'templates' in entry: return entry['templates']
        if 'all_embeddings' in entry: return entry['all_embeddings']
        if 'embedding' in entry: return [entry['embedding']]
        return []
    elif isinstance(entry, list):
        return entry
    elif isinstance(entry, np.ndarray):
        return [entry]
    return []

def is_duplicate(vec, existing_vecs, threshold=0.9999):
    # Check against all existing vectors
    if len(existing_vecs) == 0: return False
    
    # Normalize input
    vec_norm = vec / np.linalg.norm(vec)
    
    for ex in existing_vecs:
        ex_norm = ex / np.linalg.norm(ex)
        sim = np.dot(vec_norm, ex_norm)
        if sim > threshold:
            return True # Duplicate found
    return False

def merge():
    print(f"🔄 Loading databases...")
    target_db = load_pkl(TARGET_FILE)
    source_db = load_pkl(SOURCE_FILE)
    
    print(f"   Target: {len(target_db)} actors")
    print(f"   Source: {len(source_db)} actors")
    
    # Backup
    backup_path = f"{TARGET_FILE}.bak_{int(datetime.now().timestamp())}"
    shutil.copy2(TARGET_FILE, backup_path)
    print(f"💾 Backup created: {backup_path}")
    
    added_actors = 0
    added_vectors = 0
    
    for name, source_entry in source_db.items():
        source_vectors = get_vectors(source_entry)
        
        if name not in target_db:
            # New Actor
            target_db[name] = {'templates': source_vectors}
            added_actors += 1
            added_vectors += len(source_vectors)
        else:
            # Existing Actor - Merge Unique Embeddings
            target_entry = target_db[name]
            target_vectors = get_vectors(target_entry)
            
            # Ensure target is in rich format
            if not isinstance(target_entry, dict):
                target_db[name] = {'templates': target_vectors}
                target_entry = target_db[name]
            elif 'templates' not in target_entry:
                target_entry['templates'] = target_vectors
            
            # Check each source vector
            for sv in source_vectors:
                if not is_duplicate(sv, target_vectors):
                    target_entry['templates'].append(sv)
                    target_vectors.append(sv) # Update local reference for next check
                    added_vectors += 1
    
    print("-" * 30)
    print(f"✅ Merge Complete!")
    print(f"   New Actors Added: {added_actors}")
    print(f"   New Embeddings Added: {added_vectors}")
    print(f"   Total Actors Now: {len(target_db)}")
    
    save_pkl(target_db, TARGET_FILE)
    print(f"💾 Saved to {TARGET_FILE}")
    
    # Trigger Hot-Reload if Django is available
    try:
        import sys
        # Add project root to path if needed (assuming script is in root)
        sys.path.append(os.getcwd())
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_embedding_project.settings')
        import django
        django.setup()
        from django.core.cache import cache
        import time
        cache.set("recognition_db_version", time.time(), timeout=None)
        print("✅ Hot-Reload Triggered (recognition_db_version updated)")
    except Exception as e:
        print(f"⚠️ Could not trigger hot-reload (Django environment issue?): {e}")

if __name__ == "__main__":
    merge()
