import os
import django
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_embedding_project.settings')
django.setup()

from core.models import FaceGroup, Movie

def inspect():
    print("--- FACE GROUPS INSPECTION ---")
    movies = Movie.objects.all()
    for m in movies:
        print(f"Movie: {m.title}")
        groups = FaceGroup.objects.filter(movie=m)
        for g in groups:
            print(f" - Group {g.group_id}: Conf={g.avg_confidence}, Risk={g.risk_level}, Count={g.face_count}")

if __name__ == '__main__':
    inspect()
