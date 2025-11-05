import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from skimage.io import imread
from sklearn.metrics.pairwise import cosine_similarity
import os

def extract_hog_features(image_path):
    """Extract HOG features from an image"""
    try:
        print(f"Loading image: {image_path}")
        # Load image
        image = imread(image_path)
        print(f"Image shape: {image.shape}")
        
        # Convert to grayscale if it's a color image
        if image.ndim == 3:
            # Use the mean of all channels to convert to grayscale
            image = np.mean(image, axis=2)
        
        print(f"Grayscale image shape: {image.shape}")
        
        # Resize image to a consistent size
        image = resize(image, (128, 128), anti_aliasing=True)
        print(f"Resized image shape: {image.shape}")
        
        # Extract HOG features
        features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
        
        print(f"HOG features shape: {features.shape}")
        
        # Normalize the feature vector
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        raise Exception(f"Error extracting features: {str(e)}")

def compare_faces_hog(img1_path, img2_path):
    """
    Compare two face images using HOG features and return similarity score and verdict
    """
    try:
        print(f"Comparing {img1_path} and {img2_path}")
        # Extract HOG features
        features1 = extract_hog_features(img1_path)
        features2 = extract_hog_features(img2_path)
        
        # Calculate cosine similarity
        similarity_score = cosine_similarity([features1], [features2])[0][0]
        
        # Ensure the similarity is in the range [0, 1]
        similarity_score = max(0, min(1, similarity_score))
        
        print(f"Similarity score: {similarity_score}")
        
        # Determine verdict based on similarity score
        if similarity_score > 0.75:
            verdict = "Same Person / Highly Similar"
        elif 0.55 < similarity_score <= 0.75:
            verdict = "Possibly Similar"
        else:
            verdict = "Different Faces"
        
        return {
            "similarity_score": float(similarity_score),
            "verdict": verdict
        }
    except Exception as e:
        # If feature extraction fails, return error details
        print(f"Error in comparison: {str(e)}")
        return {
            "similarity_score": 0.0,
            "verdict": f"Error in processing: {str(e)}"
        }

# Test with sample images if they exist
if __name__ == "__main__":
    # Check if we have any images in the uploads directory
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        images = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) >= 2:
            img1_path = os.path.join(uploads_dir, images[0])
            img2_path = os.path.join(uploads_dir, images[1])
            result = compare_faces_hog(img1_path, img2_path)
            print(f"Test result: {result}")
        else:
            print("Not enough images in uploads directory for testing")
    else:
        print("Uploads directory does not exist")