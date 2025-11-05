import requests
import os

def test_fastapi_comparison():
    # First, check if we have images in the uploads directory
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print("Uploads directory does not exist")
        return
    
    images = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(images) < 2:
        print("Not enough images in uploads directory for testing")
        return
    
    # Test the FastAPI compare_faces endpoint
    url = "http://localhost:8000/compare_faces"
    
    # For FastAPI, we need to send the images as files
    try:
        image1_path = os.path.join(uploads_dir, images[0])
        image2_path = os.path.join(uploads_dir, images[1])
        
        with open(image1_path, 'rb') as img1, open(image2_path, 'rb') as img2:
            files = {
                'image1': (images[0], img1, 'image/png'),
                'image2': (images[1], img2, 'image/png')
            }
            
            response = requests.post(url, files=files)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error calling FastAPI: {e}")
        print("Make sure the FastAPI app is running on port 8000")

if __name__ == "__main__":
    test_fastapi_comparison()