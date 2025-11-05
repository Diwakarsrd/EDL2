import requests
import os
import json

# Test the Flask API
def test_flask_comparison():
    # First, check if we have images in the uploads directory
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print("Uploads directory does not exist")
        return
    
    images = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(images) < 2:
        print("Not enough images in uploads directory for testing")
        return
    
    # Test the compare_faces endpoint
    url = "http://localhost:5000/compare_faces"
    
    # Prepare the data
    data = {
        "image1": images[0],
        "image2": images[1]
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error calling Flask API: {e}")
        print("Make sure the Flask app is running on port 5000")

if __name__ == "__main__":
    test_flask_comparison()