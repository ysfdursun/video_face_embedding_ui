import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from django.core.files.base import ContentFile
from io import BytesIO

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from django.conf import settings

def process_and_extract_faces_stream(video_path, movie_title):
    """
    Videoyu işler, yüzleri tespit eder, diskte saklar ve işlenen kareleri stream eder.
    """
    try:
        print(f"'{movie_title}' için yüz tanıma ve stream işlemi başlıyor...")

        # Çıktı klasörünü oluştur
        unlabeled_dir = os.path.join(settings.MEDIA_ROOT, 'unlabeled_faces', movie_title)
        os.makedirs(unlabeled_dir, exist_ok=True)

        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Video dosyası açılamadı: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        face_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # --- Yüz tanımayı her 10 karede bir yap ---
            if frame_count % 10 == 0:
                faces = app.get(frame)
                if faces:
                    print(f"Kare {frame_count}: {len(faces)} adet yüz bulundu.")
                    for face_data in faces:
                        face_index += 1
                        bbox = face_data.bbox.astype(int)
                        
                        face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        if face_img.size == 0: continue
                        
                        base_filename = f"face_{face_index}"
                        jpg_filename = f"{base_filename}.jpg"
                        npy_filename = f"{base_filename}.npy"
                        
                        cv2.imwrite(os.path.join(unlabeled_dir, jpg_filename), face_img)
                        np.save(os.path.join(unlabeled_dir, npy_filename), face_data.normed_embedding)

                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # --- Her karede ilerleme yüzdesini yazdır ---
            progress = (frame_count / total_frames) * 100
            progress_text = f"Ilerleme: {progress:.2f}%"
            # Yazı için parametreler
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (255, 255, 255) # Beyaz
            thickness = 2
            # Yazının arkasına siyah bir kutu çizerek okunabilirliği artır
            (text_width, text_height), baseline = cv2.getTextSize(progress_text, font, font_scale, thickness)
            cv2.rectangle(frame, (5, frame.shape[0] - 5 - text_height - baseline), (10 + text_width, frame.shape[0] - 5), (0,0,0), -1)
            cv2.putText(frame, progress_text, (10, frame.shape[0] - 10), font, font_scale, color, thickness)

            # İşlenmiş kareyi stream için hazırla
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()
        print(f"'{movie_title}' için işlem tamamlandı.")

    except Exception as e:
        print(f"Hata: {movie_title} işlenirken bir sorun oluştu: {e}")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')


