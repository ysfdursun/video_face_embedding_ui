from django.shortcuts import render, get_object_or_404
from core.services import movie_service

def movies_dashboard(request):
    """
    Displays a grid of movies with preview cast.
    """
    page = request.GET.get('page', 1)
    search_query = request.GET.get('search', '')
    
    movies_page = movie_service.get_movies_list(page=page, search_query=search_query)
    
    return render(request, 'core/movies_dashboard.html', {
        'movies_page': movies_page,
        'search_query': search_query,
    })

def movie_detail(request, movie_id):
    """
    Displays movie details and full cast.
    """
    data = movie_service.get_movie_details(movie_id)
    if not data:
         # Handle or redirect? For now 404
         return render(request, 'core/404.html', status=404) # Assuming we will make a simplistic 404 or generic error
         
    return render(request, 'core/movie_detail.html', data)
