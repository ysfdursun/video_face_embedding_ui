import os
import cv2
import glob
from insightface.app import FaceAnalysis

# Define search grid
DET_SIZES = [(160, 160), (320, 320), (640, 640), (1280, 1280)]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

def analyze_failures():
    # Find all failed images
    image_paths = glob.glob(r'c:\Users\hamza\OneDrive\Belgeler\GitHub\video_face_embedding_ui\media\debug_failed_detections\*.jpg')
    
    if not image_paths:
        print("No failed images found.")
        return

    print(f"Found {len(image_paths)} failed images. Analyzing...")
    
    # Initialize Detector App
    app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    
    for img_path in image_paths:
        print(f"\n==================================================")
        print(f"Analyzing: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to read image.")
            continue
            
        h, w = img.shape[:2]
        print(f"Original Size: {w}x{h}")
        
        success = False
        
        for size in DET_SIZES:
            for thresh in THRESHOLDS:
                try:
                    app.prepare(ctx_id=0, det_size=size, det_thresh=thresh)
                    faces = app.get(img)
                    
                    if faces:
                        print(f"   [SUCCESS] Size={size}, Thresh={thresh} -> Found {len(faces)} faces.")
                        
                        # Print face bbox info
                        for i, face in enumerate(faces):
                            box = face.bbox.astype(int)
                            fw, fh = box[2]-box[0], box[3]-box[1]
                            print(f"      Face #{i+1}: {fw}x{fh} at {box}")
                        
                        success = True
                    # else:
                    #     print(f"   [FAIL] Size={size}, Thresh={thresh}")
                except Exception as e:
                    print(f"   [ERROR] Size={size}, Thresh={thresh}: {e}")
        
        if not success:
            print("   !!! FAILED with ALL combinations !!!")

if __name__ == '__main__':
    analyze_failures()
