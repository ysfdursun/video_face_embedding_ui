# -*- coding: utf-8 -*-
from django import forms
from .models import Movie

class VideoUploadForm(forms.Form):
    movie = forms.ModelChoiceField(
        queryset=Movie.objects.all().order_by('title'),
        label="Film Adı",
        widget=forms.Select(attrs={'class': 'form-control'}),
        empty_label="-- Film Seçin (Opsiyonel) --",
        required=False
    )
    video_file = forms.FileField(
        label="Video Dosyası",
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )

