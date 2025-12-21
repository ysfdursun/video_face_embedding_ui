#!/usr/bin/env python
import os
import sys
import django
import csv
from django.db import transaction

# Django ayarlarını yükle
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_embedding_project.settings')
django.setup()

from core.models import Actor, Movie, MovieCast

def normalize_name(name):
    """
    Adı normalize et: boşlukları kaldır, altçizgiyi boşlukla değiştir
    Örnek: "henry_fonda" -> "Henry Fonda"
    """
    name = name.replace('.', '')
    return name.lower()


def load_actors():
    """actors.txt dosyasından aktörleri yükle (Bulk Insert)"""
    actors_path = os.path.join('core', 'actors.txt')
    print(f"Aktörler yükleniyor: {actors_path}")
    
    with open(actors_path, 'r', encoding='utf-8') as f:
        # İlk satır başlık, atla
        next(f)
        file_actor_names = set()
        for line in f:
            raw_name = line.strip()
            if raw_name:
                file_actor_names.add(normalize_name(raw_name))

    # Mevcut aktörleri çek
    existing_actors = set(Actor.objects.filter(name__in=file_actor_names).values_list('name', flat=True))
    
    # Yeni eklenecekleri bul
    new_names = file_actor_names - existing_actors
    
    # Bulk create
    new_actors = [Actor(name=name) for name in new_names]
    if new_actors:
        Actor.objects.bulk_create(new_actors, batch_size=1000)
    
    total = Actor.objects.count()
    print(f"✓ Aktörler yüklendi: {len(new_actors)} yeni eklendi, {len(existing_actors)} zaten mevcut")
    print(f"  Toplam aktör sayısı: {total}")
    return total


def load_movies_and_cast():
    """movie_actors_lists.csv dosyasından filmleri ve cast'i yükle (Bulk Insert)"""
    csv_path = os.path.join('core', 'movie_actors_lists.csv')
    print(f"\nFilmler ve cast yükleniyor: {csv_path}")
    
    # Tüm aktörleri hafızaya al (ID lookup için)
    # Lowercase name -> Actor Object
    all_actors_map = {actor.name.lower(): actor for actor in Actor.objects.all()}
    print(f"Aktör veritabanında {len(all_actors_map)} aktör bulundu")

    movies_to_create = []
    cast_relations = []
    
    # Dosyayı oku ve veriyi hazırla
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader) # Header
        
        # Geçici hafıza
        file_movies = {} # title -> [actor_names]
        
        for row in csv_reader:
            if len(row) < 3: continue
            
            title = row[0].strip()
            actors_str = row[2].strip()
            
            if title and actors_str:
                file_movies[title] = [name.strip() for name in actors_str.split(',')]

    # 1. FİLMLERİ YÜKLE
    # Mevcut filmleri bul
    existing_titles = set(Movie.objects.filter(title__in=file_movies.keys()).values_list('title', flat=True))
    new_titles = set(file_movies.keys()) - existing_titles
    
    print(f"İşlenecek {len(new_titles)} yeni film var.")

    # Yeni filmleri oluştur
    new_movie_objs = [Movie(title=title) for title in new_titles]
    if new_movie_objs:
        Movie.objects.bulk_create(new_movie_objs, batch_size=1000)
        print(f"✓ {len(new_movie_objs)} film veritabanına kaydedildi.")
    
    # Şimdi tüm ilgili filmleri (eski + yeni) tekrar çekip ID'lerini alalım
    current_movies_map = {m.title: m for m in Movie.objects.filter(title__in=file_movies.keys())}
    
    # 2. CAST İLİŞKİLERİNİ YÜKLE
    # Mevcut ilişkileri çekip tekrar eklemeyelim (FilmID - ActorID pair'i)
    # Bu biraz maliyetli olabilir, o yüzden sadece yeni eklediğimiz filmler için direkt ekleyip,
    # eski filmler için kontrol yapabiliriz ya da bulk_create(ignore_conflicts=True) kullanabiliriz (Postgres'te çalışır).
    # Basitlik ve performans için: Sadece veritabanındaki ID'leri kullanarak listeyi hazırlayalım.
    
    new_cast_objects = []
    skipped_count = 0
    
    print("Cast ilişkileri hazırlanıyor...")
    
    for title, actor_names_list in file_movies.items():
        movie_obj = current_movies_map.get(title)
        if not movie_obj: continue 
        
        for actor_name_raw in actor_names_list:
            norm_name = normalize_name(actor_name_raw)
            actor_obj = all_actors_map.get(norm_name)
            
            if actor_obj:
                new_cast_objects.append(MovieCast(movie=movie_obj, actor=actor_obj))
            else:
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"  ⚠ Aktör bulunamadı: '{norm_name}' (Film: {title})")

    # Mevcut cast'i kontrol etmek yerine direkt ignore_conflicts=True ile ekleyelim
    # (Eğer unique constraint varsa). MovieCast modelinde unique constraint yoksa duplicate olabilir.
    # Genelde MovieCast modelinde class Meta: unique_together = ('movie', 'actor') olur.
    # Varsayalım ki var, ya da duplicate olmasın diye kontrol edelim.
    # Ancak o kontrol yavaşlatır. NeonDB (Postgres) için ignore_conflicts=True çok hızlıdır.
    
    if new_cast_objects:
        print(f"{len(new_cast_objects)} cast ilişkisi veritabanına yazılıyor...")
        # ignore_conflicts=True -> SQLite ve Postgres'te (farklı parametrelerle) çalışır. 
        # Django 4+ da ignore_conflicts=True parametresi var.
        MovieCast.objects.bulk_create(new_cast_objects, batch_size=2000, ignore_conflicts=True)
    
    total_movies = Movie.objects.count()
    total_cast = MovieCast.objects.count()
    
    print(f"✓ İşlem tamamlandı.")
    print(f"  Toplam film sayısı: {total_movies}")
    print(f"  Toplam cast kaydı: {total_cast}")
    print(f"  Bulunamayan aktörler: {skipped_count}")

    return total_movies, total_cast, skipped_count


def main():
    print("=" * 60)
    print("VERİ YÜKLEME İŞLEMİ (BULK OPTIMIZED)")
    print("=" * 60)
    
    with transaction.atomic():
        # Aktörleri yükle
        actor_count = load_actors()
        
        # Filmleri ve cast'i yükle
        movie_count, cast_count, skipped_count = load_movies_and_cast()
    
    print("\n" + "=" * 60)
    print("VERİ YÜKLEME İŞLEMİ TAMAMLANDI")
    print("=" * 60)
    print(f"Toplam Aktör: {actor_count}")
    print(f"Toplam Film: {movie_count}")
    print(f"Toplam Cast Üyesi: {cast_count}")
    print("=" * 60)


if __name__ == '__main__':
    main()
