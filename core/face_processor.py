# -*- coding: utf-8 -*-
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from django.conf import settings


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


def process_and_extract_faces_stream(video_path, movie_title, group_faces=True, frame_skip_extract=10, frame_skip_group=5, sim_threshold=0.45, min_face_size=80):
    """
    OPTIMIZED PIPELINE: Videoyu BİR KERE okur, yüzleri SADECE gruplandırarak kaydeder
    
    ✓ VIDEO 1 KERE OKUNUR
    ✓ FACE DETECTION 1 KERE YAPILIR
    ✓ SADECE GROUPED_FACES KLASÖRÜNE KAYDEDER (unlabeled gereksiz)
    """
    try:
        print(f"'{movie_title}' için yüz tanıma başlıyor...")
        
        # Çıktı klasörünü oluştur - SADECE grouped_faces
        grouped_dir = os.path.join(settings.MEDIA_ROOT, 'grouped_faces', movie_title)
        os.makedirs(grouped_dir, exist_ok=True)
        
        # Provider'ı al (GPU/CPU)
        providers = get_execution_providers()
        app = FaceAnalysis(name='buffalo_l', providers=providers)
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
        
        # Grouping state
        celebrities = [] if group_faces else None  # [{"rep": embedding, "count": int, "path": dir}, ...]
        
        def assign_identity(emb):
            """Embedding'i mevcut gruplarla karşılaştır."""
            if not celebrities:
                return None
            sims = [float(np.dot(emb, c["rep"])) for c in celebrities]
            best = int(np.argmax(sims))
            if sims[best] > sim_threshold:
                return best
            return None
        
        print(f"İşleme başlanıyor: Toplam {total_frames} kare\n")
        
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
                    print(f"📹 Kare {frame_count}: {len(faces)} yüz bulundu")
                    
                    for face_data in faces:
                        face_index += 1
                        bbox = face_data.bbox.astype(int)
                        
                        # Sınırları kontrol et (display için)
                        h, w = frame.shape[:2]
                        x1 = max(0, min(bbox[0], w))
                        y1 = max(0, min(bbox[1], h))
                        x2 = max(0, min(bbox[2], w))
                        y2 = max(0, min(bbox[3], h))
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Display kareye dikdörtgen çiz
                        cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_for_display, f"#{face_index}", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # --- GROUPING: Her yüzü SADECE gruplandırarak kaydet ---
                        if frame_count % frame_skip_group == 0:
                            # ALIGNED CROP: Sadece kayıt anında yap (performans için)
                            if hasattr(face_data, 'kps') and face_data.kps is not None:
                                face_img = face_align.norm_crop(frame, landmark=face_data.kps, image_size=112)
                            else:
                                face_img = frame[y1:y2, x1:x2]
                            
                            if face_img.size == 0:
                                continue
                            emb = face_data.normed_embedding
                            idx = assign_identity(emb)
                            
                            if idx is None:  # Yeni grup
                                idx = len(celebrities)
                                celeb_dir = os.path.join(grouped_dir, f"celebrity_{idx:03d}")
                                os.makedirs(celeb_dir, exist_ok=True)
                                celebrities.append({
                                    "rep": emb,
                                    "count": 0,
                                    "path": celeb_dir
                                })
                                print(f"    → Yeni grup oluşturuldu: celebrity_{idx:03d}")
                            
                            # EMA: Embedding'i güncelle (adaptif tanıma)
                            celebrities[idx]["rep"] = 0.9 * celebrities[idx]["rep"] + 0.1 * emb
                            
                            # Crop'u gruplandırılmış klasöre kaydet
                            count = celebrities[idx]["count"]
                            save_path = os.path.join(celebrities[idx]["path"], f"img_{count:04d}.jpg")
                            cv2.imwrite(save_path, face_img)
                            celebrities[idx]["count"] += 1
            
            # --- İlerleme göstergesi ---
            progress = (frame_count / total_frames) * 100
            faces_str = f"Faces: {face_index}"
            groups_str = f" | Groups: {len(celebrities)}" if group_faces and celebrities else ""
            progress_text = f"Progress: {progress:.1f}% | {faces_str}{groups_str}"
            
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
        
        # Özet
        if celebrities:
            total_saved = sum(c["count"] for c in celebrities)
            print(f"\n✓ '{movie_title}' tamamlandı:")
            print(f"  • {face_index} yüz tespit edildi")
            print(f"  • {len(celebrities)} grup oluşturuldu")
            print(f"  • {total_saved} yüz grouped_faces/ klasörüne kaydedildi")
        else:
            print(f"\n✓ '{movie_title}' tamamlandı: {face_index} yüz tespit edildi")
        
    except Exception as e:
        print(f"\n✗ Hata: {movie_title} işlenirken: {e}")
        import traceback
        traceback.print_exc()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')

        

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

