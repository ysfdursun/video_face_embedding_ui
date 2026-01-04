"""
Model Loading Utilities
=======================
Functions to load detection and recognition models.
"""

import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from .config import Config


def load_detector(det_size=None, det_thresh=None, providers=None):
    """
    Load face detection model (buffalo_sc / SCRFD).
    
    Args:
        det_size: Detection input size, default from Config
        det_thresh: Detection threshold, default from Config
        providers: Execution providers, default from Config
        
    Returns:
        FaceAnalysis app ready for inference
    """
    det_size = det_size or Config.DETECTION_SIZE
    det_thresh = det_thresh or Config.DETECTION_THRESHOLD
    providers = providers or Config.PROVIDERS
    
    print(f"📦 Loading detector: {Config.DETECTION_MODEL}")
    
    # Initialize FaceAnalysis with the detection model name
    app = FaceAnalysis(
        name=Config.DETECTION_MODEL,
        providers=providers,
        root=Config.get_insightface_root()
    )
    # Prepare with specific size and threshold
    app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
    
    print(f"   ✓ Detection model loaded (size={det_size}, thresh={det_thresh})")
    return app


def load_recognizer(providers=None):
    """
    Load face recognition model (buffalo_l / w600k_r50).
    
    Args:
        providers: Execution providers, default from Config
        
    Returns:
        Recognition model ready for inference, or None if not found
    """
    providers = providers or Config.PROVIDERS
    model_path = Config.get_recognition_model_path()
    
    print(f"📦 Loading recognizer: {Config.RECOGNITION_MODEL}")
    
    if not os.path.exists(model_path):
        print(f"   ❌ Model not found: {model_path}")
        print(f"   💡 Try running: python -c \"from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l')\"")
        return None
    
    rec_model = get_model(model_path, providers=providers)
    rec_model.prepare(ctx_id=0)
    
    print(f"   ✓ Recognition model loaded: {Config.RECOGNITION_MODEL_FILE}")
    return rec_model


def load_all_models(det_size=None, det_thresh=None, providers=None):
    """
    Load both detection and recognition models.
    
    Returns:
        tuple: (detector_app, recognizer_model)
    """
    detector = load_detector(det_size, det_thresh, providers)
    recognizer = load_recognizer(providers)
    return detector, recognizer
