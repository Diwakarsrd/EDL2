import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import json

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store features for comparison
image_features = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_simple_features(image_path):
    """Extract simple features using PIL and numpy only"""
    try:
        # Open image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to a consistent size
        image = image.resize((64, 64))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate simple statistics as features
        # Mean of each channel
        mean_r = np.mean(img_array[:,:,0])
        mean_g = np.mean(img_array[:,:,1])
        mean_b = np.mean(img_array[:,:,2])
        
        # Standard deviation of each channel
        std_r = np.std(img_array[:,:,0])
        std_g = np.std(img_array[:,:,1])
        std_b = np.std(img_array[:,:,2])
        
        # Flatten and normalize
        features = np.array([mean_r, mean_g, mean_b, std_r, std_g, std_b])
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def compare_images_simple(image1, image2):
    """Compare two images using simple features"""
    features1 = image_features.get(image1)
    features2 = image_features.get(image2)
    
    if features1 is None or features2 is None:
        return None
    
    # Cosine similarity
    dot_product = np.dot(features1, features2)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    
    if norm1 == 0 or norm2 == 0:
        cosine_similarity = 0
    else:
        cosine_similarity = dot_product / (norm1 * norm2)
    
    # Convert to percentage
    similarity_percentage = max(0, min(1, cosine_similarity)) * 100
    
    return {
        'cosine_similarity': float(cosine_similarity),
        'similarity_percentage': float(similarity_percentage)
    }

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index_vercel.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) if file.filename else 'unnamed'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract simple features
        features = extract_simple_features(filepath)
        if features is not None:
            image_features[filename] = features
        
        return jsonify({'filename': filename, 'uploaded': True})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/compare', methods=['POST'])
def compare_images():
    data = request.get_json()
    image1 = data.get('image1')
    image2 = data.get('image2')
    
    if image1 not in image_features or image2 not in image_features:
        return jsonify({'error': 'One or both images not found'})
    
    # Calculate similarity
    result = compare_images_simple(image1, image2)
    
    if result is None:
        return jsonify({'error': 'Error calculating similarity'})
    
    # Add interpretation
    similarity_percentage = result['similarity_percentage']
    if similarity_percentage > 75:
        verdict = "🟢 Same Image / Very Similar"
    elif similarity_percentage > 55:
        verdict = "🟡 Possibly Similar"
    else:
        verdict = "🔴 Different Images"
    
    result['verdict'] = verdict
    result['image1'] = image1
    result['image2'] = image2
    
    return jsonify(result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/all_images')
def list_all_images():
    # Get uploaded images
    uploaded_images = []
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            if allowed_file(filename):
                uploaded_images.append({
                    'name': filename,
                    'source': 'uploads',
                    'key': filename
                })
    
    return jsonify({
        'uploaded_images': uploaded_images,
        'dataset_images': []
    })

if __name__ == '__main__':
    app.run(debug=True)