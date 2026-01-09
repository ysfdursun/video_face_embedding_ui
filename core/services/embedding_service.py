import os
import pickle
import numpy as np
import cv2
import time
from django.conf import settings
from django.core.cache import cache
from core.config import Config
from core.model_loader import load_all_models

class EmbeddingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance.db_path = os.path.join(settings.BASE_DIR, 'embeddings_all.pkl')
            cls._instance.detector = None
            cls._instance.recognizer = None
        return cls._instance

    def _ensure_models(self):
        if self.detector is None or self.recognizer is None:
             self.detector, self.recognizer = load_all_models()

    def load_db(self):
        if not os.path.exists(self.db_path):
            return {}
        try:
            with open(self.db_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading DB: {e}")
            return {}

    def save_db(self, db):
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(db, f)
            
            # Helper to trigger Hot-Reload
            cache.set("recognition_db_version", time.time())
            print("✅ DB Saved & Hot-Reload Triggered")
            return True
        except Exception as e:
            print(f"Error saving DB: {e}")
            return False

    def get_actor_files(self, actor_name):
        """Returns set of filenames that have active embeddings for this actor."""
        db = self.load_db()
        if actor_name not in db:
            return set()
        
        data = db[actor_name]
        if isinstance(data, dict):
            return set(data.get('files', []))
        return set() # Legacy format usually has no files info

    def toggle_embedding(self, actor_name, photo_path_relative):
        """
        Toggles embedding status for a photo.
        Returns: (success, new_status, message)
        new_status: True(Active), False(Inactive)
        """
        full_path = os.path.join(settings.MEDIA_ROOT, photo_path_relative)
        filename = os.path.basename(full_path)
        
        if not os.path.exists(full_path):
            return False, False, "Fotoğraf dosyası bulunamadı"

        db = self.load_db()
        
        # Normalize actor data
        if actor_name not in db:
            db[actor_name] = {'templates': [], 'files': []}
        else:
            data = db[actor_name]
            if not isinstance(data, dict):
                # Convert legacy list to dict
                db[actor_name] = {'templates': list(data) if isinstance(data, list) else [data], 'files': []}
            elif 'templates' not in data:
                 db[actor_name] = {'templates': [], 'files': []} # Reset if malformed
        
        actor_data = db[actor_name]
        files = actor_data.get('files', [])
        templates = actor_data.get('templates', [])
        
        # Ensure lists are synced (basic check)
        if len(files) != len(templates):
            # If out of sync, we can't safely remove by index. 
            # Strategy: If we want to add, just append. If remove, relying on filename is unsafe if synced.
            # But for now, let's assume valid state or append-only if empty files.
            pass

        if filename in files:
            # REMOVE
            try:
                idx = files.index(filename)
                if idx < len(templates):
                    del templates[idx]
                files.remove(filename)
                actor_data['templates'] = templates
                actor_data['files'] = files
                self.save_db(db)
                return True, False, "Embedding veritabanından silindi"
            except ValueError:
                return False, True, "Dosya indeksi hatası"
        else:
            # ADD
            self._ensure_models()
            
            # Read Image (Unicode Safe for Windows)
            try:
                img_array = np.fromfile(full_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception:
                img = None
                
            if img is None:
                return False, False, "Görüntü okunamadı (Dosya/Unicode hatası)"
                
            # Relax detection for manual addition (Trust user)
            # Use Config value (Default: 0.1)
            original_thresh = self.detector.det_thresh
            manual_thresh = getattr(Config, 'MANUAL_ADD_THRESHOLD', 0.1)
            self.detector.det_thresh = manual_thresh
            
            try:
                faces = self.detector.get(img)
            finally:
                self.detector.det_thresh = original_thresh
                
            # Fallback: Enhance image if no face found
            if not faces:
                # Try CLAHE (Contrast Limited Adaptive Histogram Equalization)
                try:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl,a,b))
                    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                    
                    # Retry detection with VERY low threshold
                    self.detector.det_thresh = 0.05
                    faces = self.detector.get(enhanced)
                except Exception as e:
                    print(f"Fallback detection failed: {e}")
                finally:
                    self.detector.det_thresh = original_thresh

            # Fallback 2: Upscale Image (for small crops)
            if not faces:
                try:
                    h, w = img.shape[:2]
                    scale = 640 / max(h, w)
                    if scale > 1.0: # Only upscale if smaller than 640
                        upscaled = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                        self.detector.det_thresh = 0.05
                        faces_up = self.detector.get(upscaled)
                        
                        if faces_up:
                            # We need to scale back the bbox and landmarks to original image coordinates
                            face = max(faces_up, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                            face.bbox = face.bbox / scale
                            face.kps = face.kps / scale
                            faces = [face]
                except Exception as e:
                    print(f"Upscale detection failed: {e}")
                finally:
                    self.detector.det_thresh = original_thresh

            # Fallback 3: Add Padding (Zoom Out effect)
            # Useful if the face fills the entire frame (extreme close-up)
            if not faces:
                try:
                    h, w = img.shape[:2]
                    pad_h = int(h * 0.25) # Add 25% padding
                    pad_w = int(w * 0.25)
                    padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[128,128,128])
                    
                    self.detector.det_thresh = 0.05
                    faces_pad = self.detector.get(padded)
                    
                    if faces_pad:
                        # Map coordinates back
                        face = max(faces_pad, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                        face.bbox[0] -= pad_w
                        face.bbox[1] -= pad_h
                        face.bbox[2] -= pad_w
                        face.bbox[3] -= pad_h
                        face.kps[:, 0] -= pad_w
                        face.kps[:, 1] -= pad_h
                        faces = [face]
                except Exception as e:
                    print(f"Padding detection failed: {e}")
                finally:
                    self.detector.det_thresh = original_thresh

            if not faces:
                return False, False, f"Görüntüde yüz bulunamadı (Eşik: {manual_thresh}, Upscale ve Padding denenmesine rağmen)"
            
            # Use largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            
            # Extract Embedding (using code from face_recognizer logic essentially)
            M = cv2.estimateAffinePartial2D(face.kps, Config.REF_LANDMARKS, method=cv2.RANSAC)[0]
            if M is None: return False, False, "Hizalama başarısız"
            
            aligned = cv2.warpAffine(img, M, Config.ALIGNED_FACE_SIZE)
            embedding = self.recognizer.get_feat(aligned).flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0: embedding /= norm
            
            # Save
            templates.append(embedding)
            files.append(filename)
            
            actor_data['templates'] = templates
            actor_data['files'] = files
            self.save_db(db)
            
            return True, True, "Embedding başarıyla oluşturuldu ve eklendi"

    def remove_embedding(self, actor_name, photo_path_relative):
        """
        Removes embedding for a photo if it exists.
        Returns: (success, message)
        """
        full_path = os.path.join(settings.MEDIA_ROOT, photo_path_relative)
        filename = os.path.basename(full_path)
        
        db = self.load_db()
        if actor_name not in db:
            return False, "Aktör bulunamadı"
            
        actor_data = db[actor_name]
        if not isinstance(actor_data, dict):
            return False, "Veri formatı eski"
            
        files = actor_data.get('files', [])
        templates = actor_data.get('templates', [])
        
        if filename in files:
            try:
                idx = files.index(filename)
                if idx < len(templates):
                    del templates[idx]
                files.remove(filename)
                actor_data['templates'] = templates
                actor_data['files'] = files
                self.save_db(db)
                return True, "Embedding silindi"
            except Exception as e:
                return False, f"Hata: {str(e)}"
        
        return True, "Embedding zaten yoktu"

    def delete_actor_embeddings(self, actor_name):
        """
        Removes all embeddings for an actor from the database.
        Returns: (success, message)
        """
        db = self.load_db()
        if actor_name in db:
            try:
                del db[actor_name]
                self.save_db(db)
                return True, f"'{actor_name}' veritabanından tamamen silindi."
            except Exception as e:
                return False, f"Silme hatası: {str(e)}"
        return True, "Kayıt zaten yoktu."

embedding_service = EmbeddingService()
