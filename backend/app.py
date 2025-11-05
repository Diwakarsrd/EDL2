import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import uvicorn
from skimage.feature import hog
from skimage.transform import resize
from skimage.io import imread
from sklearn.metrics.pairwise import cosine_similarity
import uuid

app = FastAPI(title="Face Similarity Detector")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

def extract_hog_features(image_path):
    """Extract HOG features from an image"""
    try:
        # Load image
        image = imread(image_path)
        
        # Convert to grayscale if it's a color image
        if image.ndim == 3:
            # Use the mean of all channels to convert to grayscale
            image = np.mean(image, axis=2)
        
        # Resize image to a consistent size
        image = resize(image, (128, 128), anti_aliasing=True)
        
        # Extract HOG features
        features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
        
        # Normalize the feature vector
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")

def compare_faces_hog(img1_path, img2_path):
    """
    Compare two face images using HOG features and return similarity score and verdict
    """
    try:
        # Extract HOG features
        features1 = extract_hog_features(img1_path)
        features2 = extract_hog_features(img2_path)
        
        # Calculate cosine similarity
        similarity_score = cosine_similarity([features1], [features2])[0][0]
        
        # Ensure the similarity is in the range [0, 1]
        similarity_score = max(0, min(1, similarity_score))
        
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
        return {
            "similarity_score": 0.0,
            "verdict": f"Error in processing: {str(e)}"
        }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🧠 AI Face Similarity Detector</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f7fb;
                color: #333;
            }
            header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #4361ee, #3f37c9);
                color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            h1 {
                margin: 0;
                font-size: 2.5rem;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .upload-section {
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
            }
            .upload-box {
                flex: 1;
                border: 2px dashed #4361ee;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                background-color: white;
            }
            .upload-box h3 {
                margin-top: 0;
                color: #4361ee;
            }
            .preview-container {
                margin-top: 10px;
                display: flex;
                justify-content: center;
            }
            .preview-container img {
                max-width: 100%;
                max-height: 200px;
                border-radius: 5px;
            }
            input[type="file"] {
                width: 100%;
                margin: 10px 0;
                padding: 10px;
            }
            button {
                background-color: #4361ee;
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 18px;
                font-weight: bold;
                transition: background-color 0.3s;
                display: block;
                margin: 20px auto;
                width: 80%;
            }
            button:hover {
                background-color: #3f37c9;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            #result {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                display: none;
            }
            .similarity-score {
                font-size: 1.5rem;
                font-weight: bold;
                margin: 10px 0;
            }
            .verdict {
                font-size: 1.2rem;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .verdict.high {
                background-color: #d1f0dd;
                color: #2a9d8f;
            }
            .verdict.medium {
                background-color: #fff3cd;
                color: #856404;
            }
            .verdict.low {
                background-color: #f8d7da;
                color: #721c24;
            }
            .progress-container {
                width: 100%;
                background-color: #e9ecef;
                border-radius: 10px;
                margin: 20px 0;
                height: 30px;
                overflow: hidden;
            }
            .progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #ff6b6b, #ffd166, #06d6a0);
                border-radius: 10px;
                transition: width 0.5s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            }
            .image-comparison {
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
            }
            .image-comparison img {
                max-width: 45%;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            #loading {
                text-align: center;
                display: none;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #4361ee;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            .error-message {
                background-color: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
                display: none;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>🧠 AI Face Similarity Detector</h1>
            <p>Upload two face images to check how similar they are</p>
        </header>
        
        <div class="container">
            <div class="upload-section">
                <div class="upload-box">
                    <h3>Face Image 1</h3>
                    <input type="file" id="image1" accept="image/*" onchange="previewImage(event, 'preview1')">
                    <div class="preview-container">
                        <img id="preview1" style="display: none;">
                    </div>
                </div>
                
                <div class="upload-box">
                    <h3>Face Image 2</h3>
                    <input type="file" id="image2" accept="image/*" onchange="previewImage(event, 'preview2')">
                    <div class="preview-container">
                        <img id="preview2" style="display: none;">
                    </div>
                </div>
            </div>
            
            <button id="checkButton" onclick="checkSimilarity()" disabled>Check Similarity</button>
            
            <div id="loading">
                <div class="spinner"></div>
                <p>Analyzing face similarity...</p>
            </div>
            
            <div id="errorMessage" class="error-message"></div>
            
            <div id="result">
                <h2>Similarity Result</h2>
                <div class="image-comparison">
                    <img id="resultImage1">
                    <img id="resultImage2">
                </div>
                <div class="similarity-score">Similarity Score: <span id="score">0.00</span></div>
                <div class="progress-container">
                    <div class="progress-bar" id="progressBar" style="width: 0%">0%</div>
                </div>
                <div class="verdict" id="verdict">-</div>
            </div>
        </div>

        <script>
            let image1File = null;
            let image2File = null;
            
            function previewImage(event, previewId) {
                const file = event.target.files[0];
                const preview = document.getElementById(previewId);
                
                if (file) {
                    const reader = new FileReader();
                    
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                        
                        // Store files for upload
                        if (previewId === 'preview1') {
                            image1File = file;
                        } else {
                            image2File = file;
                        }
                        
                        // Enable button if both images are selected
                        if (image1File && image2File) {
                            document.getElementById('checkButton').disabled = false;
                        }
                    }
                    
                    reader.readAsDataURL(file);
                } else {
                    preview.style.display = 'none';
                }
            }
            
            async function checkSimilarity() {
                if (!image1File || !image2File) {
                    showError('Please select both images');
                    return;
                }
                
                // Show loading indicator
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                document.getElementById('errorMessage').style.display = 'none';
                document.getElementById('checkButton').disabled = true;
                
                try {
                    const formData = new FormData();
                    formData.append('image1', image1File);
                    formData.append('image2', image2File);
                    
                    const response = await fetch('/compare_faces', {
                        method: 'POST',
                        body: formData
                    });
                    
                    // Check if response is JSON
                    const contentType = response.headers.get('content-type');
                    if (!contentType || !contentType.includes('application/json')) {
                        const errorText = await response.text();
                        throw new Error(`Server error: ${response.status} - ${errorText.substring(0, 200)}...`);
                    }
                    
                    const result = await response.json();
                    
                    // Hide loading indicator
                    document.getElementById('loading').style.display = 'none';
                    
                    // Check if result contains error
                    if (result.verdict && result.verdict.includes('Error')) {
                        showError(result.verdict);
                        return;
                    }
                    
                    // Display results
                    document.getElementById('score').textContent = result.similarity_score.toFixed(2);
                    document.getElementById('progressBar').style.width = (result.similarity_score * 100) + '%';
                    document.getElementById('progressBar').textContent = Math.round(result.similarity_score * 100) + '%';
                    
                    // Set verdict text and class
                    const verdictElement = document.getElementById('verdict');
                    verdictElement.textContent = result.verdict;
                    
                    // Set verdict color based on similarity
                    verdictElement.className = 'verdict';
                    if (result.similarity_score > 0.75) {
                        verdictElement.classList.add('high');
                    } else if (result.similarity_score > 0.55) {
                        verdictElement.classList.add('medium');
                    } else {
                        verdictElement.classList.add('low');
                    }
                    
                    // Set result images
                    document.getElementById('resultImage1').src = URL.createObjectURL(image1File);
                    document.getElementById('resultImage2').src = URL.createObjectURL(image2File);
                    
                    // Show result
                    document.getElementById('result').style.display = 'block';
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('checkButton').disabled = false;
                    showError('Error comparing faces: ' + error.message);
                }
            }
            
            function showError(message) {
                const errorElement = document.getElementById('errorMessage');
                errorElement.textContent = message;
                errorElement.style.display = 'block';
                document.getElementById('checkButton').disabled = false;
            }
        </script>
    </body>
    </html>
    """

@app.post("/compare_faces")
async def compare_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    """
    Compare two face images and return similarity score and verdict
    """
    # Initialize paths
    image1_path = None
    image2_path = None
    
    try:
        # Validate file types
        if image1.content_type is None or not image1.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="First file must be an image")
        
        if image2.content_type is None or not image2.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Second file must be an image")
        
        # Generate unique filenames to avoid conflicts
        image1_filename = f"{uuid.uuid4()}_{image1.filename}"
        image2_filename = f"{uuid.uuid4()}_{image2.filename}"
        
        image1_path = os.path.join(UPLOADS_DIR, image1_filename)
        image2_path = os.path.join(UPLOADS_DIR, image2_filename)
        
        # Read image1
        contents1 = await image1.read()
        with open(image1_path, "wb") as f:
            f.write(contents1)
        
        # Read image2
        contents2 = await image2.read()
        with open(image2_path, "wb") as f:
            f.write(contents2)
        
        # Compare faces using HOG features
        result = compare_faces_hog(image1_path, image2_path)
        
        # Clean up temporary files
        if os.path.exists(image1_path):
            os.remove(image1_path)
        if os.path.exists(image2_path):
            os.remove(image2_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up any files that might have been created
        try:
            if image1_path and os.path.exists(image1_path):
                os.remove(image1_path)
            if image2_path and os.path.exists(image2_path):
                os.remove(image2_path)
        except:
            pass
            
        error_msg = f"Error processing images: {str(e)}"
        return JSONResponse(
            content={
                "similarity_score": 0.0,
                "verdict": error_msg
            },
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)