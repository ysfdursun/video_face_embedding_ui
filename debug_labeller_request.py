import requests
import os
import pprint

# Configuration
URL = 'http://127.0.0.1:8000/api/labeller/analyze/'
IMAGE_PATH = r'c:\Users\hamza\OneDrive\Belgeler\GitHub\video_face_embedding_ui\media\labeled_faces\aamir_khan\foto_1.jpg'

def debug_labeller():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return
        
    size = os.path.getsize(IMAGE_PATH)
    print(f"Testing Labeller API with: {IMAGE_PATH} (Size: {size} bytes)")
    
    try:
        with open(IMAGE_PATH, 'rb') as f:
            files = {'image': f}
            # Test with default settings first
            data = {'det_thresh': 0.5, 'min_blur': 80}
            
            print(f"Sending POST request to {URL}...")
            response = requests.post(URL, files=files, data=data)
            
            print(f"Status Code: {response.status_code}")
            print("-" * 40)
            try:
                pprint.pprint(response.json())
            except:
                print("Response Text (Not JSON):")
                print(response.text)
            print("-" * 40)
                
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    debug_labeller()
