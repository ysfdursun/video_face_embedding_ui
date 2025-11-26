from django import forms

class VideoUploadForm(forms.Form):
    title = forms.CharField(
        max_length=255, 
        label="Film Adı",
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'İşlem için bir başlık girin'})
    )
    video_file = forms.FileField(
        label="Video Dosyası",
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )

