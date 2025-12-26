# -*- coding: utf-8 -*-
"""
Face Processor Module - Enhanced with Quality Scoring

Integrates:
- FaceQualityScorer: Blur, illumination, pose scoring
- DuplicateDetector: pHash and embedding similarity
- FaceDetection model: Persistent quality metrics storage
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from django.conf import settings
import time

# Quality and duplicate detection modules
from core.quality.face_quality import get_quality_scorer
from core.quality.duplicate_detector import get_duplicate_detector


def get_execution_providers():
    """GPU varsa CUDA kullan, yoksa CPU fallback'e geç."""
    try:
        import onnxruntime
        available_providers = onnxruntime.get_available_providers()
        print(f"Mevcut providers: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            print("✓ CUDA GPU desteği kullanılacak")
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            print("⚠ CUDA bulunamadı, CPU kullanılacak")
            return ['CPUExecutionProvider']
    except Exception as e:
        print(f"Provider kontrolünde hata: {e}, CPU kullanılacak")
        return ['CPUExecutionProvider']


def process_and_extract_faces_stream(video_path, movie_title, group_faces=True):
    """
    OPTIMIZED PIPELINE: Videoyu BİR KERE okur, yüzleri SADECE gruplandırarak kaydeder
    Uses Global Settings from DB if available.
    """
    
    # Cache key setup
    from django.core.cache import cache
    from core.utils.file_utils import get_safe_filename
    from core.models import FaceRecognitionSettings, Movie, FaceGroup, FaceDetection
    
    # Initialize quality and duplicate checkers
    quality_scorer = get_quality_scorer()
    duplicate_detector = get_duplicate_detector()
    
    # --- GET SETTINGS ---
    try:
        settings_obj = FaceRecognitionSettings.get_settings()
        sim_threshold = settings_obj.grouping_threshold
        min_face_size = settings_obj.min_face_size
        frame_skip_extract = settings_obj.frame_skip_extract
        frame_skip_group = settings_obj.frame_skip_group
        gpu_enabled = settings_obj.gpu_enabled
        quality_threshold = settings_obj.quality_threshold
        redundancy_threshold = settings_obj.redundancy_threshold
    except Exception as e:
        print(f"Error loading settings, using defaults: {e}")
        sim_threshold = 0.45
        min_face_size = 80
        frame_skip_extract = 10
        frame_skip_group = 5
        gpu_enabled = True
        quality_threshold = 0.50
        redundancy_threshold = 0.85
    
    # Apply thresholds to quality modules
    quality_scorer.MIN_QUALITY_SCORE = quality_threshold
    duplicate_detector.REDUNDANT_THRESHOLD = redundancy_threshold

    safe_movie_title = get_safe_filename(movie_title)
    cache_key = f"processing_status_{safe_movie_title}"
    
    try:
        print(f"'{movie_title}' için yüz tanıma başlıyor (Threshold: {sim_threshold}, GPU: {gpu_enabled})...")
        
        # Movie objesini al veya oluştur (FaceGroup için gerekli)
        # Try exact match first, then safe filename match
        movie_obj = Movie.objects.filter(title=movie_title).first()
        if not movie_obj:
            movie_obj = Movie.objects.filter(title=safe_movie_title).first()
        if not movie_obj:
            movie_obj, _ = Movie.objects.get_or_create(title=safe_movie_title)
        
        # Çıktı klasörünü oluştur - safe filename kullan!
        grouped_dir = os.path.join(settings.MEDIA_ROOT, 'grouped_faces', safe_movie_title)
        
        # CLEANUP: Önceki verileri sil (Reprocess = Fresh Start)
        if os.path.exists(grouped_dir):
            import shutil
            try:
                print(f"🧹 Önceki veriler temizleniyor: {grouped_dir}")
                shutil.rmtree(grouped_dir)
                # DB Cleanup
                FaceGroup.objects.filter(movie=movie_obj).delete()
            except Exception as e:
                print(f"⚠ Temizlik uyarısı: {e}")
                
        os.makedirs(grouped_dir, exist_ok=True)
        
        # Status: RUNNING
        cache.set(cache_key, "running", timeout=3600)
        
        # Provider'ı al (GPU/CPU)
        providers = get_execution_providers()
        if not gpu_enabled:
             providers = ['CPUExecutionProvider']

        insightface_root = os.path.join(os.path.expanduser("~"), ".insightface")

        # [1/2] Detection: buffalo_sc (SCRFD-2.5G içerir)
        print("\n[1/2] Loading SCRFD-2.5G detector (via buffalo_sc)...")
        app = FaceAnalysis(
            name='buffalo_sc',
            root=insightface_root,
            providers=providers
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Face alignment için gerekli
        from insightface.utils import face_align
        face_aligner = face_align
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Video dosyası açılamadı: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        face_index = 0
        skipped_quality = 0  # Track quality-filtered faces
        skipped_duplicate = 0  # Track duplicate-filtered faces
        
        # Grouping state with enhanced tracking
        celebrities = [] if group_faces else None
        # Track recent hashes for duplicate detection within video
        recent_hashes = []  # List of (hash, group_idx) tuples
        MAX_RECENT_HASHES = 500  # Sliding window size  
        
        def assign_identity(emb):
            """Embedding'i mevcut gruplarla karşılaştır (Adaptive Threshold)."""
            if not celebrities:
                return None, 0.0
            
            sims = [float(np.dot(emb, c["rep"])) for c in celebrities]
            best = int(np.argmax(sims))
            score = sims[best]
            
            # --- ADAPTIVE THRESHOLD LOGIC ---
            # Grup büyüdükçe eşik değerini artır (Daha seçici ol)
            # Penalty: Her yüz için +0.001, Maksimum +0.10
            # Örn: 50 yüzlük grup için -> 0.45 + 0.05 = 0.50 olur
            current_count = celebrities[best]["sim_count"]
            penalty = min(0.10, current_count * 0.001)
            dynamic_threshold = sim_threshold + penalty
            
            if score > dynamic_threshold:
                return best, score
            
            # Eğer skor, dinamik eşiğin altındaysa ama base eşiğin üstündeyse
            # Bunu opsiyonel olarak loglayabiliriz ama performans için sessiz geçiyoruz.
            return None, score
        
        print(f"İşleme başlanıyor: Toplam {total_frames} kare\n")
        
        fps_start_time = time.time()
        fps_counter = 0
        fps_display = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_for_display = frame.copy()  # Stream için görüntü
            
            # --- EXTRACT: Her N karede yüz tespit ---
            if frame_count % frame_skip_extract == 0:
                faces = app.get(frame)
                if faces:
                    # Log less frequently to avoid console spam if needed, but keeping for now
                    # print(f"📹 Kare {frame_count}: {len(faces)} yüz bulundu")
                    
                    for face_data in faces:
                        face_index += 1
                        bbox = face_data.bbox.astype(int)
                        det_score = float(face_data.det_score) if hasattr(face_data, 'det_score') else 1.0
                        
                        # Sınırları kontrol et (display için)
                        h, w = frame.shape[:2]
                        x1 = max(0, min(bbox[0], w))
                        y1 = max(0, min(bbox[1], h))
                        x2 = max(0, min(bbox[2], w))
                        y2 = max(0, min(bbox[3], h))
                        
                        # Min Face Size Check
                        if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
                            continue
                        
                        # Crop face for quality analysis - use ALIGNED face
                        # norm_crop returns 112x112 aligned face
                        if hasattr(face_data, 'kps') and face_data.kps is not None:
                            face_img = face_aligner.norm_crop(frame, face_data.kps)
                        else:
                            # Fallback to bbox crop if no landmarks
                            face_img = frame[y1:y2, x1:x2]
                        
                        if face_img is None or face_img.size == 0:
                            continue
                        
                        # === QUALITY SCORING ===
                        landmarks = face_data.kps if hasattr(face_data, 'kps') else None
                        quality_result = quality_scorer.get_combined_score(
                            face_img, 
                            landmarks,
                            det_score
                        )
                        
                        # Quality gating: Skip low-quality faces
                        if not quality_result['is_valid']:
                            skipped_quality += 1
                            # Draw red box for rejected faces
                            cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            continue
                        
                        # === DUPLICATE DETECTION (pHash) ===
                        face_hash = duplicate_detector.compute_phash(face_img)
                        
                        # Check against recent hashes
                        is_duplicate = False
                        for prev_hash, _ in recent_hashes:
                            if duplicate_detector.is_near_duplicate(face_hash, prev_hash):
                                is_duplicate = True
                                break
                        
                        if is_duplicate:
                            skipped_duplicate += 1
                            # Draw yellow box for duplicates
                            cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (0, 255, 255), 1)
                            continue
                        
                        # Get embedding for grouping and redundancy check
                        emb = face_data.normed_embedding
                        
                        # === EMBEDDING REDUNDANCY CHECK ===
                        # Check if this face is too similar to existing faces in any group
                        is_redundant = False
                        for celeb in celebrities:
                            if celeb.get("embeddings"):
                                is_red, sim = duplicate_detector.is_redundant(emb, celeb["embeddings"][-20:])  # Check last 20
                                if is_red:
                                    is_redundant = True
                                    break
                        
                        if is_redundant:
                            skipped_duplicate += 1
                            # Draw orange box for embedding redundant
                            cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (0, 165, 255), 1)
                            continue
                        
                        # Display green box for accepted faces
                        cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Show quality score on frame
                        q_text = f"Q:{quality_result['quality_score']:.2f}"
                        cv2.putText(frame_for_display, q_text, (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        
                        # === GROUPING ===
                        # emb already extracted above for redundancy check
                        idx, match_score = assign_identity(emb)
                        
                        if idx is None:  # New group
                            idx = len(celebrities)
                            celeb_dir = os.path.join(grouped_dir, f"celebrity_{idx:03d}")
                            os.makedirs(celeb_dir, exist_ok=True)
                            celebrities.append({
                                "rep": emb,
                                "count": 0,
                                "path": celeb_dir,
                                "total_sim": 0.0,
                                "sim_count": 0,
                                "total_quality": 0.0,
                                "quality_count": 0,
                                "embeddings": [emb],  # Store for multi-template
                                "group_id": f"celebrity_{idx:03d}"
                            })
                            print(f"    → New group created: celebrity_{idx:03d}")
                            match_score = 1.0
                        else:
                            # Store embedding for multi-template (limited to 50)
                            if len(celebrities[idx].get("embeddings", [])) < 50:
                                if "embeddings" not in celebrities[idx]:
                                    celebrities[idx]["embeddings"] = []
                                celebrities[idx]["embeddings"].append(emb)
                        
                        # Update stats
                        if match_score > 0:
                            celebrities[idx]["total_sim"] += match_score
                            celebrities[idx]["sim_count"] += 1
                        
                        # Update quality stats
                        celebrities[idx]["total_quality"] = celebrities[idx].get("total_quality", 0) + quality_result['quality_score']
                        celebrities[idx]["quality_count"] = celebrities[idx].get("quality_count", 0) + 1
                        
                        # EMA update for representative embedding
                        celebrities[idx]["rep"] = 0.9 * celebrities[idx]["rep"] + 0.1 * emb
                        
                        # Save face image
                        count = celebrities[idx]["count"]
                        save_path = os.path.join(celebrities[idx]["path"], f"img_{count:04d}.jpg")
                        cv2.imwrite(save_path, face_img)
                        celebrities[idx]["count"] += 1
                        
                        # Update recent hashes (sliding window)
                        recent_hashes.append((face_hash, idx))
                        if len(recent_hashes) > MAX_RECENT_HASHES:
                            recent_hashes.pop(0)
            
            # --- İlerleme göstergesi ---
            progress = (frame_count / total_frames) * 100
            # Ensure 100% is displayed at end
            
            faces_str = f"Faces: {face_index}"
            groups_str = f" | Groups: {len(celebrities)}" if group_faces and celebrities else ""
            skip_str = f" | Skip Q:{skipped_quality} D:{skipped_duplicate}"
            
            # FPS Calculation
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_end_time = time.time()
                fps_display = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            
            progress_text = f"FPS: {fps_display:.1f} | Progress: {progress:.1f}% | {faces_str}{groups_str}{skip_str}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(progress_text, font, 0.7, 2)
            cv2.rectangle(frame_for_display, (5, frame_for_display.shape[0] - 5 - text_height - baseline),
                         (10 + text_width, frame_for_display.shape[0] - 5), (0, 0, 0), -1)
            cv2.putText(frame_for_display, progress_text, (10, frame_for_display.shape[0] - 10),
                       font, 0.7, (255, 255, 255), 2)
            
            # --- Stream için encode ---
            is_success, buffer = cv2.imencode(".jpg", frame_for_display)
            if is_success:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()
        
        # --- POST-PROCESSING: SAVE GROUP STATS TO DB ---
        if celebrities:
            print(f"\n📊 Processing Summary:")
            print(f"   Total faces detected: {face_index}")
            print(f"   Skipped (quality): {skipped_quality}")
            print(f"   Skipped (duplicate): {skipped_duplicate}")
            print(f"   Groups created: {len(celebrities)}")
            print("\nSaving group stats to database...")
            
            for c in celebrities:
                try:
                    # Calculate average confidence
                    avg_conf = 1.0
                    if c["sim_count"] > 0:
                        avg_conf = c["total_sim"] / c["sim_count"]
                    
                    # Calculate average quality
                    avg_quality = 0.0
                    if c.get("quality_count", 0) > 0:
                        avg_quality = c["total_quality"] / c["quality_count"]
                    
                    # Update or Create FaceGroup
                    fg, created = FaceGroup.objects.get_or_create(
                        movie=movie_obj,
                        group_id=c["group_id"]
                    )
                    fg.face_count = c["count"]
                    fg.total_faces = c["count"]
                    fg.avg_confidence = avg_conf
                    fg.avg_quality = avg_quality
                    
                    # Store representative embedding
                    if c.get("embeddings"):
                        # Use mean embedding as representative
                        rep_emb = np.mean(c["embeddings"], axis=0)
                        rep_emb = rep_emb / np.linalg.norm(rep_emb)
                        fg.set_representative_embedding(rep_emb)
                    
                    fg.update_risk_level()
                    
                except Exception as e:
                    print(f"Error saving FaceGroup stats for {c['group_id']}: {e}")

        # Set COMPLETED status explicitly here before potentially exiting
        print(f"Video loop finished for {movie_title}")
        
    except Exception as e:
        print(f"\n✗ Hata: {movie_title} işlenirken: {e}")
        import traceback
        traceback.print_exc()
        cache.set(cache_key, "error", timeout=3600)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
            
    finally:
        # Guarantee status update
        # Check if we didn't error out already
        current_status = cache.get(cache_key)
        if current_status != "error":
             print(f"Setting COMPLETED status for {movie_title} (Key: {cache_key})")
             cache.set(cache_key, "completed", timeout=3600)
             # Invalidate Group List Cache so UI sees new groups immediately
             cache.delete(f"groups_list_{safe_movie_title}")
             
        # Cleanup if needed
        if 'cap' in locals() and cap.isOpened():
            cap.release()

        

def process_and_group_faces(video_path, movie_title, frame_skip=5, sim_threshold=0.45, min_face_size=80):
    """
    DEPRECATED - Eski fonksiyon (geriye uyumluluk için)
    
    ⚠ YENİ KOD YAZARKEN process_and_extract_faces_stream() KULLANIN!
    Sebep: İki videoyu ayrı ayrı işlemek yerine BİR KERE işler, 2x daha hızlıdır.
    """
    print(f"⚠️  UYARI: process_and_group_faces() deprecated!")
    print(f"   Yerine: process_and_extract_faces_stream(group_faces=True) kullanın")
    print(f"   İşlem yine de devam ediyor...\n")
    
    out_dir = os.path.join(settings.MEDIA_ROOT, "grouped_faces", movie_title)
    os.makedirs(out_dir, exist_ok=True)

    providers = get_execution_providers()
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    celebrities = []

    def assign_identity(emb):
        if not celebrities:
            return None
        sims = [float(np.dot(emb, c["rep"])) for c in celebrities]
        best = int(np.argmax(sims))
        if sims[best] > sim_threshold:
            return best
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Video açılamadı: {video_path}")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip:
            frame_id += 1
            continue

        faces = app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            h, w = frame.shape[:2]
            
            x1 = max(0, min(bbox[0], w))
            y1 = max(0, min(bbox[1], h))
            x2 = max(0, min(bbox[2], w))
            y2 = max(0, min(bbox[3], h))

            if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            emb = face.normed_embedding
            idx = assign_identity(emb)

            if idx is None:
                idx = len(celebrities)
                celebrities.append({"rep": emb, "count": 0})
                os.makedirs(os.path.join(out_dir, f"celebrity_{idx:03d}"), exist_ok=True)

            celebrities[idx]["rep"] = 0.9 * celebrities[idx]["rep"] + 0.1 * emb

            count = celebrities[idx]["count"]
            save_path = os.path.join(out_dir, f"celebrity_{idx:03d}", f"img_{count:04d}.jpg")
            cv2.imwrite(save_path, crop)
            celebrities[idx]["count"] += 1

        frame_id += 1

    cap.release()
    print(f"✓ '{movie_title}': {len(celebrities)} grup oluşturuldu -> {out_dir}\n")
    return out_dir

