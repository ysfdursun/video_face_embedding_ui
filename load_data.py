#!/usr/bin/env python
import os
import sys
import django
import csv

# Django ayarlarını yükle
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_embedding_project.settings')
django.setup()

from core.models import Actor, Movie, MovieCast

def normalize_name(name):
    """
    Adı normalize et: boşlukları kaldır, altçizgiyi boşlukla değiştir
    Örnek: "henry_fonda" -> "Henry Fonda"
    """
    # replace . to ""
    name = name.replace('.', '')
    # # Altçizgileri boşlukla değiştir
    # name = name.replace('_', ' ')
    # # Her kelimenin baş harfini büyüt
    # name = ' '.join(word.capitalize() for word in name.split())
    return name.lower()


def load_actors():
    """actors.txt dosyasından aktörleri yükle"""
    actors_path = os.path.join('core', 'actors.txt')
    
    print(f"Aktörler yükleniyor: {actors_path}")
    
    with open(actors_path, 'r', encoding='utf-8') as f:
        # İlk satır başlık, atla
        next(f)
        
        loaded_count = 0
        existing_count = 0
        
        for line in f:
            actor_name_raw = line.strip()
            if not actor_name_raw:
                continue
            
            actor_name = normalize_name(actor_name_raw)
            
            actor, created = Actor.objects.get_or_create(
                name=actor_name,
                defaults={'name': actor_name}
            )
            
            if created:
                loaded_count += 1
            else:
                existing_count += 1
    
    total = Actor.objects.count()
    print(f"✓ Aktörler yüklendi: {loaded_count} yeni eklendi, {existing_count} zaten mevcut")
    print(f"  Toplam aktör sayısı: {total}")
    return total


def load_movies_and_cast():
    """movie_actors_lists.csv dosyasından filmleri ve cast'i yükle"""
    csv_path = os.path.join('core', 'movie_actors_lists.csv')
    
    print(f"\nFilmler ve cast yükleniyor: {csv_path}")
    
    movies_created = 0
    movies_existing = 0
    cast_loaded = 0
    cast_skipped = 0
    
    all_actors = {actor.name.lower(): actor for actor in Actor.objects.all()}
    
    print(f"Aktör veritabanında {len(all_actors)} aktör bulundu")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        # İlk satır başlık, atla
        header = next(csv_reader)
        
        for row in csv_reader:
            if len(row) < 3:
                continue
            
            movie_title = row[0].strip()
            actors_str = row[2].strip()
            
            if not movie_title or not actors_str:
                continue
            
            # Filmi oluştur veya al
            movie, movie_created = Movie.objects.get_or_create(title=movie_title)
            
            if movie_created:
                movies_created += 1
            else:
                movies_existing += 1
            
            # Cast üyelerini işle
            actor_names = [name.strip() for name in actors_str.split(',')]
            
            for actor_name_raw in actor_names:
                actor_name_normalized = normalize_name(actor_name_raw)
                
                # Aktörü bul
                actor = None
                
                # Önce exact match dene
                for db_actor_name, db_actor in all_actors.items():
                    if db_actor_name == actor_name_normalized.lower():
                        actor = db_actor
                        break
                
                if actor:
                    # MovieCast kaydını oluştur
                    cast_member, created = MovieCast.objects.get_or_create(
                        movie=movie,
                        actor=actor
                    )
                    if created:
                        cast_loaded += 1
                else:
                    cast_skipped += 1
                    # Biraz bilgi ver
                    if cast_skipped <= 20:  # İlk 20 eksik aktörü göster
                        print(f"  ⚠ Aktör bulunamadı: '{actor_name_normalized}' (Film: {movie_title})")
    
    total_movies = Movie.objects.count()
    total_cast = MovieCast.objects.count()
    
    print(f"✓ Filmler yüklendi: {movies_created} yeni eklendi, {movies_existing} zaten mevcut")
    print(f"  Toplam film sayısı: {total_movies}")
    print(f"✓ Cast üyeleri yüklendi: {cast_loaded} eklendi, {cast_skipped} bulunamadı")
    print(f"  Toplam cast kaydı: {total_cast}")
    
    return total_movies, total_cast, cast_skipped


def main():
    print("=" * 60)
    print("VERİ YÜKLEME İŞLEMİ BAŞLANIYOR")
    print("=" * 60)
    
    # Mevcut veriyi temizle (opsiyonel)
    print("\nMevcut veriler kontrol ediliyor...")
    print(f"  Aktörler: {Actor.objects.count()}")
    print(f"  Filmler: {Movie.objects.count()}")
    print(f"  Cast Üyeleri: {MovieCast.objects.count()}")
    
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
    if skipped_count > 0:
        print(f"⚠ Bulunamayan Cast Üyeleri: {skipped_count}")
    print("=" * 60)


if __name__ == '__main__':
    main()
