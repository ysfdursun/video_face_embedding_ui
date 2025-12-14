from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.urls import reverse
import os
import shutil
import threading
from .forms import VideoUploadForm
from .face_processor import process_and_extract_faces_stream, process_and_group_faces


def get_safe_filename(name):
    """Remove special chars and spaces to build a safe filename/folder name."""
    return "".join([c for c in name if c.isalpha() or c.isdigit() or c == ' ']).rstrip().replace(' ', '_')


def welcome(request):
    """Landing page with upload/label options."""
    return render(request, 'core/welcome.html')


def home(request):
    """
    Main page. Handles video upload and lists uploaded videos and pending unlabeled faces.
    """
    video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
    os.makedirs(video_dir, exist_ok=True)

    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = request.FILES['video_file']
            title = form.cleaned_data['title']

            safe_title = get_safe_filename(title)
            _, file_ext = os.path.splitext(video_file.name)
            final_filename = f"{safe_title}{file_ext}"

            video_path = os.path.join(video_dir, final_filename)
            with open(video_path, 'wb+') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)

            return redirect('core:processing_page', movie_filename=final_filename)
    else:
        form = VideoUploadForm()

    videos = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]

    unlabeled_faces_dir = os.path.join(settings.MEDIA_ROOT, 'unlabeled_faces')
    pending_faces_count = 0
    if os.path.exists(unlabeled_faces_dir):
        for movie_folder in os.listdir(unlabeled_faces_dir):
            movie_path = os.path.join(unlabeled_faces_dir, movie_folder)
            if os.path.isdir(movie_path):
                pending_faces_count += len([f for f in os.listdir(movie_path) if f.endswith('.jpg')])

    return render(request, 'core/home.html', {
        'form': form,
        'videos': videos,
        'pending_faces_count': pending_faces_count,
    })


def processing_page(request, movie_filename):
    """Video processing screen that shows the stream."""
    movie_title = os.path.splitext(movie_filename)[0]
    return render(request, 'core/processing.html', {'movie_title': movie_title, 'movie_filename': movie_filename})


def stream_video_processing(request, movie_filename):
    """Runs the processing generator and starts background grouping."""
    video_path = os.path.join(settings.MEDIA_ROOT, 'videos', movie_filename)
    movie_title = os.path.splitext(movie_filename)[0]

    def _background_grouping():
        try:
            out_dir = os.path.join(settings.MEDIA_ROOT, "grouped_faces", movie_title)
            if os.path.exists(out_dir) and any(os.scandir(out_dir)):
                print(f"Grouping skipped: {out_dir} already exists.")
                return
            print(f"Grouping started: {movie_title}")
            process_and_group_faces(video_path, movie_title)
        except Exception as e:
            print(f"Grouping error: {e}")

    threading.Thread(target=_background_grouping, daemon=True).start()

    try:
        stream = process_and_extract_faces_stream(video_path, movie_title)
        return StreamingHttpResponse(stream, content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Stream error: {e}")
        return redirect('core:home')


def list_unlabeled_faces(request):
    """List all unlabeled faces as a gallery."""
    unlabeled_dir = os.path.join(settings.MEDIA_ROOT, 'unlabeled_faces')
    os.makedirs(unlabeled_dir, exist_ok=True)

    unlabeled_faces = []
    for movie_folder in sorted(os.listdir(unlabeled_dir)):
        movie_path = os.path.join(unlabeled_dir, movie_folder)
        if os.path.isdir(movie_path):
            for filename in sorted(os.listdir(movie_path)):
                if filename.endswith('.jpg'):
                    unlabeled_faces.append({
                        'path': os.path.join('unlabeled_faces', movie_folder, filename),
                        'movie_title': movie_folder
                    })

    context = {
        'unlabeled_faces': unlabeled_faces,
        'unlabeled_faces_count': len(unlabeled_faces),
    }
    return render(request, 'core/list_unlabeled.html', context)


def label_all_faces(request):
    """Iterate through all unlabeled faces and allow labeling/discarding."""
    unlabeled_dir = os.path.join(settings.MEDIA_ROOT, 'unlabeled_faces')
    labeled_dir = os.path.join(settings.MEDIA_ROOT, 'labeled_faces')
    os.makedirs(unlabeled_dir, exist_ok=True)
    os.makedirs(labeled_dir, exist_ok=True)

    # always collect full list of pending faces
    all_unlabeled_faces = []
    for movie_folder in sorted(os.listdir(unlabeled_dir)):
        movie_path = os.path.join(unlabeled_dir, movie_folder)
        if os.path.isdir(movie_path):
            for filename in sorted(os.listdir(movie_path)):
                if filename.endswith('.jpg'):
                    all_unlabeled_faces.append(os.path.join('unlabeled_faces', movie_folder, filename))

    if request.method == 'POST':
        face_path_relative = request.POST.get('face_path')
        action = request.POST.get('action')

        next_face_path = None
        try:
            current_index = all_unlabeled_faces.index(face_path_relative)
            if current_index + 1 < len(all_unlabeled_faces):
                next_face_path = all_unlabeled_faces[current_index + 1]
        except ValueError:
            pass

        if face_path_relative and action:
            face_path_full = os.path.join(settings.MEDIA_ROOT, face_path_relative)
            base_name = os.path.splitext(face_path_full)[0]
            jpg_path = f"{base_name}.jpg"
            npy_path = f"{base_name}.npy"

            if action == 'discard':
                if os.path.exists(jpg_path):
                    os.remove(jpg_path)
                if os.path.exists(npy_path):
                    os.remove(npy_path)

            elif action == 'save':
                cast_name = request.POST.get('assigned_cast_name')
                if cast_name:
                    safe_cast_name = get_safe_filename(cast_name)
                    new_cast_dir = os.path.join(labeled_dir, safe_cast_name)
                    os.makedirs(new_cast_dir, exist_ok=True)

                    file_count = len(os.listdir(new_cast_dir))
                    new_jpg_path = os.path.join(new_cast_dir, f"face_{file_count + 1}.jpg")
                    new_npy_path = os.path.join(new_cast_dir, f"face_{file_count + 1}.npy")

                    if os.path.exists(jpg_path):
                        shutil.move(jpg_path, new_jpg_path)
                    if os.path.exists(npy_path):
                        shutil.move(npy_path, new_npy_path)

        if next_face_path:
            return redirect(f"{reverse('core:label_all_faces')}?face_path={next_face_path}")
        else:
            return redirect('core:list_unlabeled_faces')

    if not all_unlabeled_faces:
        return redirect('core:list_unlabeled_faces')

    current_face_path = request.GET.get('face_path')
    if current_face_path:
        current_face_path = os.path.normpath(current_face_path)
        if current_face_path not in all_unlabeled_faces:
            current_face_path = all_unlabeled_faces[0]
    else:
        current_face_path = all_unlabeled_faces[0]

    next_face_path_for_button = None
    try:
        current_index = all_unlabeled_faces.index(current_face_path)
        if current_index + 1 < len(all_unlabeled_faces):
            next_face_path_for_button = all_unlabeled_faces[current_index + 1]
    except ValueError:
        pass

    try:
        movie_title_for_face = current_face_path.split(os.sep)[1]
    except IndexError:
        movie_title_for_face = "Unknown Movie"

    dir_cast_list = [d.replace('_', ' ') for d in os.listdir(labeled_dir) if os.path.isdir(os.path.join(labeled_dir, d))]

    predefined_cast_list = []
    try:
        actors_file_path = os.path.join(settings.BASE_DIR, 'core', 'actors.txt')
        with open(actors_file_path, 'r', encoding='utf-8') as f:
            predefined_cast_list = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Warning: 'core/actors.txt' not found.")

    combined_cast_list = sorted(list(set(dir_cast_list + predefined_cast_list)))

    context = {
        'face_path': current_face_path,
        'next_face_path': next_face_path_for_button,
        'movie_title': movie_title_for_face,
        'cast_list': combined_cast_list,
        'post_url': reverse('core:label_all_faces'),
    }
    return render(request, 'core/label_face_filesystem.html', context)

