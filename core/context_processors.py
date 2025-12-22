from core.models import FaceRecognitionSettings

def global_settings(request):
    """
    Expose global settings (GPU enabled, etc.) to all templates.
    """
    try:
        settings_obj = FaceRecognitionSettings.get_settings()
        return {'global_settings': settings_obj}
    except Exception:
        # Fallback if DB is not ready or migration issue
        return {'global_settings': None}
