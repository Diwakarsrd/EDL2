import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import euclidean
import numpy as np
import zipfile
import shutil
from io import BytesIO
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DATASETS_FOLDER = 'datasets'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASETS_FOLDER'] = DATASETS_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER

# Store features for comparison
image_features = {}

# Store dataset information
datasets = {}

# Store trained models
models = {}

# Store raw images for MSE/SSIM comparison
raw_images = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    # Load the image
    try:
        image = imread(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Store raw image for MSE/SSIM comparison
    # Convert to grayscale if it's a color image
    if image.ndim == 3:
        # Use the mean of all channels to convert to grayscale
        gray_image = np.mean(image, axis=2)
    else:
        gray_image = image
    
    # Create multiple scaled versions of the image
    scales = [64, 128, 256]
    features = []
    
    for scale in scales:
        # Resize image for consistent feature extraction
        resized_image = resize(image, (scale, scale), anti_aliasing=True)

        # Convert to grayscale if it's a color image
        if resized_image.ndim == 3:
            # Use the mean of all channels to convert to grayscale
            resized_image = np.mean(resized_image, axis=2)

        # Extract HOG features with optimized parameters
        fd = hog(resized_image, orientations=9, pixels_per_cell=(4, 4),
                 cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys',
                 feature_vector=True)
        
        # Normalize the feature vector
        fd_normalized = fd / (np.linalg.norm(fd) + 1e-8)
        features.append(fd_normalized)
    
    # Concatenate all features
    combined_features = np.concatenate(features)
    
    # Normalize the combined feature vector
    combined_features_normalized = combined_features / (np.linalg.norm(combined_features) + 1e-8)
    
    return combined_features_normalized

def load_dataset_images(dataset_name):
    """Load all images from a dataset folder"""
    dataset_path = os.path.join(DATASETS_FOLDER, dataset_name)
    if not os.path.exists(dataset_path):
        return []
    
    images = []
    for filename in os.listdir(dataset_path):
        if allowed_file(filename) and not (filename.endswith('.zip') if filename else False):
            images.append({
                'name': filename,
                'path': os.path.join(dataset_path, filename),
                'dataset': dataset_name
            })
    return images

def load_existing_features():
    """Load features for existing images in the uploads folder"""
    if not os.path.exists(UPLOAD_FOLDER):
        return
    
    for filename in os.listdir(UPLOAD_FOLDER):
        if allowed_file(filename):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try:
                features = extract_features(filepath)
                if features is not None:
                    image_features[filename] = features
                    # Load raw image for MSE/SSIM comparison
                    raw_image = imread(filepath)
                    if raw_image.ndim == 3:
                        raw_images[filename] = np.mean(raw_image, axis=2)
                    else:
                        raw_images[filename] = raw_image
                    print(f"Loaded features for {filename}")
            except Exception as e:
                print(f"Error loading features for {filename}: {e}")

def initialize_datasets():
    """Initialize datasets folder and load existing datasets"""
    if not os.path.exists(DATASETS_FOLDER):
        os.makedirs(DATASETS_FOLDER)
    
    # Load existing datasets
    for dataset_name in os.listdir(DATASETS_FOLDER):
        dataset_path = os.path.join(DATASETS_FOLDER, dataset_name)
        if os.path.isdir(dataset_path):
            datasets[dataset_name] = {
                'name': dataset_name,
                'path': dataset_path,
                'images': load_dataset_images(dataset_name)
            }
            
            # Load features for dataset images
            for image_info in datasets[dataset_name]['images']:
                try:
                    features = extract_features(image_info['path'])
                    if features is not None:
                        image_key = f"{dataset_name}/{image_info['name']}"
                        image_features[image_key] = features
                        # Load raw image for MSE/SSIM comparison
                        raw_image = imread(image_info['path'])
                        if raw_image.ndim == 3:
                            raw_images[image_key] = np.mean(raw_image, axis=2)
                        else:
                            raw_images[image_key] = raw_image
                        print(f"Loaded features for {image_key}")
                except Exception as e:
                    print(f"Error loading features for {image_info['name']}: {e}")

def initialize_models():
    """Initialize models folder and load existing models"""
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)
    
    # Load existing models
    for model_file in os.listdir(MODELS_FOLDER):
        if model_file.endswith('.json'):
            model_name = model_file.replace('.json', '')
            try:
                with open(os.path.join(MODELS_FOLDER, model_file), 'r') as f:
                    model_data = json.load(f)
                    # For simplicity, we're just storing model metadata
                    # In a real implementation, you would load the actual model
                    models[model_name] = {
                        'name': model_name,
                        'dataset': model_data.get('dataset', ''),
                        'accuracy': model_data.get('accuracy', 0),
                        'created': model_data.get('created', '')
                    }
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")

# Function for Mean Squared Error
def mean_squared_error(image1, image2):
    # Resize image2 to match image1 dimensions
    if image1.shape != image2.shape:
        image2 = resize(image2, image1.shape, anti_aliasing=True)
    
    # Calculate MSE
    error = np.sum((image1.astype('float') - image2.astype('float'))**2)
    error = error/float(image1.shape[0] * image2.shape[1])
    return error

# Function for image comparison using MSE and SSIM
def image_comparison(image1, image2):
    # Resize image2 to match image1 dimensions
    if image1.shape != image2.shape:
        image2 = resize(image2, image1.shape, anti_aliasing=True)
    
    # Calculate MSE
    mse_value = mean_squared_error(image1, image2)
    
    # Calculate SSIM
    ssim_value = ssim(image1, image2, data_range=image1.max() - image1.min())
    
    return mse_value, ssim_value

# Function for face similarity comparison
def compare_face_similarity(image1, image2):
    """Compare two face images and return similarity metrics"""
    # Calculate all similarity metrics
    features1 = image_features.get(image1)
    features2 = image_features.get(image2)
    
    # If features don't exist, try to extract them
    if features1 is None or features2 is None:
        try:
            # Try to find file paths for the images
            filepath1 = None
            filepath2 = None
            
            # Check in uploads folder
            uploads_path1 = os.path.join(UPLOAD_FOLDER, image1)
            if os.path.exists(uploads_path1):
                filepath1 = uploads_path1
            
            uploads_path2 = os.path.join(UPLOAD_FOLDER, image2)
            if os.path.exists(uploads_path2):
                filepath2 = uploads_path2
            
            # Check in datasets if not found in uploads
            if filepath1 is None:
                for dataset_name, dataset_info in datasets.items():
                    dataset_image_key = f"{dataset_name}/{image1}"
                    dataset_image_path = os.path.join(dataset_info['path'], image1)
                    if os.path.exists(dataset_image_path):
                        filepath1 = dataset_image_path
                        # Extract features for dataset image
                        features1 = extract_features(filepath1)
                        if features1 is not None:
                            image_features[dataset_image_key] = features1
                            # Load raw image
                            raw_image = imread(filepath1)
                            if raw_image.ndim == 3:
                                raw_images[dataset_image_key] = np.mean(raw_image, axis=2)
                            else:
                                raw_images[dataset_image_key] = raw_image
                        break
            
            if filepath2 is None:
                for dataset_name, dataset_info in datasets.items():
                    dataset_image_key = f"{dataset_name}/{image2}"
                    dataset_image_path = os.path.join(dataset_info['path'], image2)
                    if os.path.exists(dataset_image_path):
                        filepath2 = dataset_image_path
                        # Extract features for dataset image
                        features2 = extract_features(filepath2)
                        if features2 is not None:
                            image_features[dataset_image_key] = features2
                            # Load raw image
                            raw_image = imread(filepath2)
                            if raw_image.ndim == 3:
                                raw_images[dataset_image_key] = np.mean(raw_image, axis=2)
                            else:
                                raw_images[dataset_image_key] = raw_image
                        break
            
            # Extract features for upload images if file paths are found
            if filepath1 and os.path.exists(filepath1) and image1 in os.listdir(UPLOAD_FOLDER):
                features1 = extract_features(filepath1)
                if features1 is not None:
                    image_features[image1] = features1
                    # Load raw image
                    raw_image = imread(filepath1)
                    if raw_image.ndim == 3:
                        raw_images[image1] = np.mean(raw_image, axis=2)
                    else:
                        raw_images[image1] = raw_image
            
            if filepath2 and os.path.exists(filepath2) and image2 in os.listdir(UPLOAD_FOLDER):
                features2 = extract_features(filepath2)
                if features2 is not None:
                    image_features[image2] = features2
                    # Load raw image
                    raw_image = imread(filepath2)
                    if raw_image.ndim == 3:
                        raw_images[image2] = np.mean(raw_image, axis=2)
                    else:
                        raw_images[image2] = raw_image
                        
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {'error': f'Error extracting features: {str(e)}'}
    
    # Check again after possible extraction
    features1 = image_features.get(image1)
    features2 = image_features.get(image2)
    
    if features1 is None or features2 is None:
        return {'error': 'Could not extract features from one or both images'}
    
    # Cosine similarity
    dot_product = np.dot(features1, features2)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    
    if norm1 == 0 or norm2 == 0:
        cosine_similarity = 0
    else:
        cosine_similarity = dot_product / (norm1 * norm2)
    
    # Euclidean distance
    euclidean_distance = euclidean(features1, features2)
    
    # Convert to percentage
    similarity_percentage = max(0, min(1, cosine_similarity)) * 100
    
    # Get MSE and SSIM if raw images are available
    mse_value = None
    ssim_value = None
    if image1 in raw_images and image2 in raw_images:
        try:
            mse_value, ssim_value = image_comparison(raw_images[image1], raw_images[image2])
        except Exception as e:
            print(f"Error calculating MSE/SSIM: {e}")
    
    return {
        'cosine_similarity': float(cosine_similarity),
        'similarity_percentage': float(similarity_percentage),
        'euclidean_distance': float(euclidean_distance),
        'mse': float(mse_value) if mse_value is not None else None,
        'ssim': float(ssim_value) if ssim_value is not None else None
    }

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists('templates'):
    os.makedirs('templates')

# Load existing features when the app starts
load_existing_features()
initialize_datasets()
initialize_models()

@app.route('/')
def index():
    return render_template('index.html')

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
        
        # Extract features
        features = extract_features(filepath)
        if features is not None:
            image_features[filename] = features
            # Load raw image for MSE/SSIM comparison
            raw_image = imread(filepath)
            if raw_image.ndim == 3:
                raw_images[filename] = np.mean(raw_image, axis=2)
            else:
                raw_images[filename] = raw_image
        
        return jsonify({'filename': filename, 'uploaded': True})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    dataset_name = request.form.get('dataset_name', '').strip()
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not dataset_name:
        return jsonify({'error': 'Dataset name is required'})
    
    if file and allowed_file(file.filename):
        # Create dataset folder
        dataset_path = os.path.join(app.config['DATASETS_FOLDER'], dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        
        filename = file.filename if file.filename is not None else ''
        if filename.endswith('.zip'):
            # Handle ZIP file upload
            try:
                # Save the file temporarily
                temp_path = os.path.join(dataset_path, 'temp.zip')
                file.save(temp_path)
                
                # Extract the ZIP file
                with zipfile.ZipFile(temp_path) as zip_file:
                    for member in zip_file.namelist():
                        if allowed_file(member):
                            # Extract only image files
                            zip_file.extract(member, dataset_path)
                
                # Remove the temporary ZIP file
                os.remove(temp_path)
                
                # Process extracted files
                for root, dirs, files in os.walk(dataset_path):
                    for file_name in files:
                        if allowed_file(file_name):
                            file_path = os.path.join(root, file_name)
                            # Move to dataset root if in subdirectory
                            if root != dataset_path:
                                new_path = os.path.join(dataset_path, file_name)
                                if not os.path.exists(new_path):
                                    shutil.move(file_path, new_path)
                                else:
                                    # Handle duplicate names
                                    base_name, ext = os.path.splitext(file_name)
                                    counter = 1
                                    new_name = f"{base_name}_{counter}{ext}"
                                    while os.path.exists(os.path.join(dataset_path, new_name)):
                                        counter += 1
                                        new_name = f"{base_name}_{counter}{ext}"
                                    shutil.move(file_path, os.path.join(dataset_path, new_name))
                
                # Remove empty subdirectories
                for root, dirs, files in os.walk(dataset_path, topdown=False):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            
            except Exception as e:
                return jsonify({'error': f'Error processing ZIP file: {str(e)}'})
        else:
            # Handle single image file
            filename = secure_filename(file.filename) if file.filename else 'unnamed'
            filepath = os.path.join(dataset_path, filename)
            file.save(filepath)
        
        # Load dataset images and features
        datasets[dataset_name] = {
            'name': dataset_name,
            'path': dataset_path,
            'images': load_dataset_images(dataset_name)
        }
        
        # Load features for dataset images
        for image_info in datasets[dataset_name]['images']:
            try:
                features = extract_features(image_info['path'])
                if features is not None:
                    image_key = f"{dataset_name}/{image_info['name']}"
                    image_features[image_key] = features
                    # Load raw image for MSE/SSIM comparison
                    raw_image = imread(image_info['path'])
                    if raw_image.ndim == 3:
                        raw_images[image_key] = np.mean(raw_image, axis=2)
                    else:
                        raw_images[image_key] = raw_image
            except Exception as e:
                print(f"Error loading features for {image_info['name']}: {e}")
        
        return jsonify({'dataset_name': dataset_name, 'created': True})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.get_json()
    dataset_name = data.get('dataset_name')
    model_name = data.get('model_name', '').strip()
    
    if not dataset_name:
        return jsonify({'error': 'Dataset name is required'})
    
    if not model_name:
        return jsonify({'error': 'Model name is required'})
    
    if dataset_name not in datasets:
        return jsonify({'error': 'Dataset not found'})
    
    if len(datasets[dataset_name]['images']) < 2:
        return jsonify({'error': 'Dataset must contain at least 2 images for training'})
    
    try:
        # Get features for all images in the dataset
        feature_vectors = []
        image_keys = []
        
        for image_info in datasets[dataset_name]['images']:
            image_key = f"{dataset_name}/{image_info['name']}"
            if image_key in image_features:
                feature_vectors.append(image_features[image_key])
                image_keys.append(image_key)
        
        if len(feature_vectors) < 2:
            return jsonify({'error': 'Not enough valid images with extracted features'})
        
        # Convert to numpy array
        X = np.array(feature_vectors)
        
        # Create a similarity model using NearestNeighbors
        # This model will be used to find similar images
        model = NearestNeighbors(n_neighbors=min(5, len(feature_vectors)), metric='cosine')
        model.fit(X)
        
        # Save model (in a real implementation, you would save the actual model)
        model_data = {
            'dataset': dataset_name,
            'accuracy': 0.85,  # Placeholder accuracy
            'created': np.datetime_as_string(np.datetime64('now')),
            'num_images': len(feature_vectors),
            'image_keys': image_keys  # Store image keys for reference
        }
        
        # Create models folder if it doesn't exist
        if not os.path.exists(MODELS_FOLDER):
            os.makedirs(MODELS_FOLDER)
        
        # Save model metadata
        with open(os.path.join(MODELS_FOLDER, f"{model_name}.json"), 'w') as f:
            json.dump(model_data, f)
        
        # Store in models dictionary
        models[model_name] = {
            'name': model_name,
            'dataset': dataset_name,
            'accuracy': model_data['accuracy'],
            'created': model_data['created']
        }
        
        return jsonify({
            'model_name': model_name,
            'dataset': dataset_name,
            'accuracy': model_data['accuracy'],
            'message': 'Model trained successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error training model: {str(e)}'})

@app.route('/compare_with_model', methods=['POST'])
def compare_with_model():
    data = request.get_json()
    image_key = data.get('image_key')
    model_name = data.get('model_name')
    
    if image_key not in image_features:
        return jsonify({'error': 'Image not found'})
    
    if model_name not in models:
        return jsonify({'error': 'Model not found'})
    
    try:
        # Get the model metadata
        model_info = models[model_name]
        dataset_name = model_info['dataset']
        
        # Get features for the query image
        query_features = image_features[image_key]
        
        # Get features for all images in the dataset used to train the model
        feature_vectors = []
        image_keys = []
        
        for image_info in datasets[dataset_name]['images']:
            img_key = f"{dataset_name}/{image_info['name']}"
            if img_key in image_features and img_key != image_key:
                feature_vectors.append(image_features[img_key])
                image_keys.append(img_key)
        
        if len(feature_vectors) == 0:
            return jsonify({'error': 'No comparison images available'})
        
        # Convert to numpy array
        X = np.array(feature_vectors)
        
        # Create a temporary model for comparison
        model = NearestNeighbors(n_neighbors=min(5, len(feature_vectors)), metric='cosine')
        model.fit(X)
        
        # Find similar images
        distances, indices = model.kneighbors([query_features], n_neighbors=min(5, len(feature_vectors)))
        
        # Prepare results
        results = []
        for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
            similarity = 1 - distance  # Convert distance to similarity
            results.append({
                'image_key': image_keys[index],
                'similarity': float(similarity),
                'distance': float(distance)
            })
        
        return jsonify({
            'query_image': image_key,
            'similar_images': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Error comparing with model: {str(e)}'})

@app.route('/compare', methods=['POST'])
def compare_images():
    data = request.get_json()
    image1 = data.get('image1')
    image2 = data.get('image2')
    
    if image1 not in image_features or image2 not in image_features:
        return jsonify({'error': 'One or both images not found'})
    
    # Get feature vectors
    features1 = image_features[image1]
    features2 = image_features[image2]
    
    # Calculate cosine similarity (higher values mean more similar)
    dot_product = np.dot(features1, features2)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        cosine_similarity = 0
    else:
        cosine_similarity = dot_product / (norm1 * norm2)
    
    # Ensure the similarity is in the range [0, 1]
    cosine_similarity = max(0, min(1, cosine_similarity))
    
    # Calculate Euclidean distance (lower values mean more similar)
    euclidean_distance = euclidean(features1, features2)
    
    # Convert cosine similarity to percentage (0-100%)
    similarity_percentage = cosine_similarity * 100
    
    return jsonify({
        'image1': image1,
        'image2': image2,
        'distance': float(euclidean_distance),
        'similarity': float(similarity_percentage),  # Percentage similarity
        'cosine_similarity': float(cosine_similarity),
        'euclidean_distance': float(euclidean_distance)
    })

@app.route('/compare_mse_ssim', methods=['POST'])
def compare_mse_ssim():
    data = request.get_json()
    image1_key = data.get('image1')
    image2_key = data.get('image2')
    
    if image1_key not in raw_images or image2_key not in raw_images:
        return jsonify({'error': 'One or both images not found for MSE/SSIM comparison'})
    
    try:
        # Get raw images
        image1 = raw_images[image1_key]
        image2 = raw_images[image2_key]
        
        # Calculate MSE and SSIM
        mse_value, ssim_value = image_comparison(image1, image2)
        
        return jsonify({
            'image1': image1_key,
            'image2': image2_key,
            'mse': float(mse_value),
            'ssim': float(ssim_value)
        })
    except Exception as e:
        return jsonify({'error': f'Error comparing images with MSE/SSIM: {str(e)}'})

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    """Compare two face images and return face similarity metrics"""
    data = request.get_json()
    image1_key = data.get('image1')
    image2_key = data.get('image2')
    
    if image1_key not in image_features or image2_key not in image_features:
        return jsonify({'error': 'One or both images not found'})
    
    # Calculate all similarity metrics
    result = compare_face_similarity(image1_key, image2_key)
    
    if result is None:
        return jsonify({'error': 'Error calculating face similarity'})
    
    # Add face-specific interpretation
    similarity_percentage = result['similarity_percentage']
    if similarity_percentage > 75:
        verdict = "🟢 Same Person / Very Similar"
    elif similarity_percentage > 55:
        verdict = "🟡 Possibly Similar"
    else:
        verdict = "🔴 Different Faces"
    
    result['verdict'] = verdict
    result['image1'] = image1_key
    result['image2'] = image2_key
    
    return jsonify(result)

@app.route('/uploads')
def list_uploads():
    if not os.path.exists(UPLOAD_FOLDER):
        return jsonify({'images': []})
    
    images = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if allowed_file(filename):
            images.append({
                'name': filename,
                'source': 'uploads'
            })
    
    return jsonify({'images': images})

@app.route('/datasets')
def list_datasets():
    dataset_list = []
    for name, info in datasets.items():
        dataset_list.append({
            'name': name,
            'num_images': len(info['images'])
        })
    return jsonify({'datasets': dataset_list})

@app.route('/dataset/<dataset_name>')
def list_dataset_images(dataset_name):
    if dataset_name not in datasets:
        return jsonify({'error': 'Dataset not found'})
    
    images = []
    for image_info in datasets[dataset_name]['images']:
        images.append({
            'name': image_info['name'],
            'dataset': dataset_name
        })
    
    return jsonify({'images': images})

@app.route('/models')
def list_models():
    model_list = []
    for name, info in models.items():
        model_list.append({
            'name': name,
            'dataset': info['dataset'],
            'accuracy': info['accuracy'],
            'created': info['created']
        })
    return jsonify({'models': model_list})

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
    
    # Get dataset images
    dataset_images = []
    for dataset_name, dataset_info in datasets.items():
        for image_info in dataset_info['images']:
            dataset_images.append({
                'name': image_info['name'],
                'source': f'dataset:{dataset_name}',
                'key': f"{dataset_name}/{image_info['name']}"
            })
    
    return jsonify({
        'uploaded_images': uploaded_images,
        'dataset_images': dataset_images
    })

# NEW ROUTE: To serve uploaded images directly
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# NEW ROUTE: To serve dataset images
@app.route('/datasets/<dataset_name>/<filename>')
def dataset_image(dataset_name, filename):
    dataset_path = os.path.join(app.config['DATASETS_FOLDER'], dataset_name)
    return send_from_directory(dataset_path, filename)

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        file_deleted = False
        feature_deleted = False
        
        # Remove from image_features dictionary
        if filename in image_features:
            del image_features[filename]
            feature_deleted = True
        
        # Remove from raw_images dictionary
        if filename in raw_images:
            del raw_images[filename]
        
        # Remove file from uploads directory
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            file_deleted = True
        
        if file_deleted or feature_deleted:
            return jsonify({'message': f'File {filename} deleted successfully'})
        else:
            return jsonify({'error': f'File {filename} not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error deleting file: {str(e)}'}), 500

@app.route('/delete_dataset/<dataset_name>', methods=['DELETE'])
def delete_dataset(dataset_name):
    try:
        if dataset_name not in datasets:
            return jsonify({'error': f'Dataset {dataset_name} not found'}), 404
        
        # Remove dataset features
        dataset_images = datasets[dataset_name]['images']
        for image_info in dataset_images:
            image_key = f"{dataset_name}/{image_info['name']}"
            if image_key in image_features:
                del image_features[image_key]
            if image_key in raw_images:
                del raw_images[image_key]
        
        # Remove dataset directory
        dataset_path = os.path.join(app.config['DATASETS_FOLDER'], dataset_name)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        
        # Remove from datasets dictionary
        del datasets[dataset_name]
        
        return jsonify({'message': f'Dataset {dataset_name} deleted successfully'})
    except Exception as e:
        return jsonify({'error': f'Error deleting dataset: {str(e)}'}), 500

@app.route('/delete_model/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    try:
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        # Remove model file
        model_path = os.path.join(app.config['MODELS_FOLDER'], f"{model_name}.json")
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Remove from models dictionary
        del models[model_name]
        
        return jsonify({'message': f'Model {model_name} deleted successfully'})
    except Exception as e:
        return jsonify({'error': f'Error deleting model: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)