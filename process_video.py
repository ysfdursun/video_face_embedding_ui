import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os

# Yüz analiz modelini hazırlama
# providers=['CPUExecutionProvider'] -> Sadece CPU kullanarak çalıştırır. Eğer CUDA destekli bir GPU'nuz varsa bu satırı silebilir veya ['CUDAExecutionProvider'] yapabilirsiniz.
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# İşlenecek video dosyasının yolu
# TODO: Buraya kendi video dosyanızın yolunu ekleyin
video_path = 'tenet.mp4'

# Çıktıların kaydedileceği ana klasör
output_dir = 'output_faces_tenet'
os.makedirs(output_dir, exist_ok=True)

# Video yakalama objesi
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Hata: Video dosyası açılamadı: {video_path}")
    exit()

all_embeddings = []
face_images = []
frame_count = 0
detected_faces = []  # Son tespit edilen yüzleri saklamak için

print("Video işleniyor... (Görüntü penceresinde 'q' tuşuna basarak çıkabilirsiniz)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display_frame = frame.copy()

    # Performans için her 5 karede bir yüz tespiti yap
    if frame_count % 5 == 0:
        faces = app.get(frame)
        detected_faces = faces  # Son tespit edilen yüzleri her zaman güncelle
        if len(faces) > 0:
            print(f"Kare {frame_count}: {len(faces)} adet yüz bulundu.")
            for face in faces:
                # Embedding'i al ve listeye ekle
                all_embeddings.append(face.normed_embedding)
                
                # Yüzün olduğu bölgeyi kırp
                bbox = face.bbox.astype(int)
                # Kırpma işlemi için orijinal 'frame' kullanılıyor
                face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                face_images.append(face_img)

    # Her karede, son tespit edilen yüzlerin kutularını çiz
    if len(detected_faces) > 0:
        for face in detected_faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # İşlenmiş kareyi ekranda göster
    cv2.imshow('Video', display_frame)

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Video işleme tamamlandı.")

if len(all_embeddings) > 0:
    print(f"Toplam {len(all_embeddings)} adet yüz embedding'i çıkarıldı.")
    
    # Embedding'leri numpy dizisine dönüştür
    embeddings_array = np.array(all_embeddings)

    # DBSCAN kullanarak kümeleme yap (cosine mesafesi ile)
    # eps değeri, iki örnek arasındaki maksimum mesafeyi belirler (0 ile 1 arası).
    # min_samples, bir noktanın çekirdek nokta olarak kabul edilmesi için komşuluğunda olması gereken minimum örnek sayısıdır.
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity

    # Cosine mesafesi = 1 - Cosine benzerliği
    # Yani, 0.5'lik bir 'eps' değeri, 0.5 ve üzeri benzerliğe sahip yüzleri gruplamaya çalışır.
    db = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    clusters = db.fit_predict(embeddings_array)

    unique_labels = np.unique(clusters)
    print(f"Toplam {len(unique_labels) - (1 if -1 in unique_labels else 0)} farklı kişi kümesi bulundu (gürültü hariç).")

    # Her kümeyi (kişiyi) ayrı bir klasöre kaydet
    for label in unique_labels:
        if label == -1:  # Gürültü noktalarını (kümelenemeyen yüzler) atla
            print("Gürültü olarak işaretlenen yüzler atlanıyor.")
            continue

        person_dir = os.path.join(output_dir, f"person_{label}")
        os.makedirs(person_dir, exist_ok=True)
        
        # Bu kümeye ait yüzlerin indekslerini al
        indices = np.where(clusters == label)[0]

        print(f"Kişi {label} için {len(indices)} adet yüz kaydediliyor...")
        for i, idx in enumerate(indices):
            try:
                # Yüz resmini kaydet
                face_filename = os.path.join(person_dir, f"face_{i:04d}.jpg")
                cv2.imwrite(face_filename, face_images[idx])
            except Exception as e:
                print(f"Hata: Yüz {idx} kaydedilemedi: {e}")
else:
    print("Hiç yüz bulunamadığı için kümeleme yapılmadı.")

print("İşlem tamamlandı. Yüzler 'output_faces' klasörüne kaydedildi.")
