# Flask Face Image Similarity Application

This is a Flask web application for comparing face images and determining their similarity using computer vision techniques.

## Features
- Upload two face images for comparison
- Analyze similarity using HOG (Histogram of Oriented Gradients) features
- Display similarity score and verdict (Same Person / Possibly Similar / Different Faces)

## Requirements
- Python 3.8+
- Flask
- scikit-image
- scikit-learn
- numpy
- scipy
- Pillow

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python app.py
```

Then open your browser to http://localhost:5000

## How It Works
The application uses HOG feature extraction to convert face images into feature vectors, then compares these vectors using cosine similarity to determine how similar the faces are.