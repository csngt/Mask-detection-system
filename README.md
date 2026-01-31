# Face Mask Detection System

This project is a Face Mask Detection web application built using Deep Learning, Flask, OpenCV, and TensorFlow.
It detects whether a person is wearing a mask correctly, wearing a mask incorrectly, or not wearing a mask.

The system supports both real-time webcam detection and image upload-based prediction.

## Project Features

- Real-time face mask detection using webcam
- Image upload and prediction
- Custom CNN model trained from scratch
- Confidence score displayed with prediction
- Flask-based web interface

## Classes Detected

The model classifies faces into the following three categories:

- Incorrectly Worn Mask
- With Mask
- Without Mask

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Flask
- NumPy
- HTML and CSS

## Project Structure

Mask Detection/
├── dataset/              # Training dataset (not included)
├── uploads/              # Uploaded images
├── templates/
│   └── index.html
├── app.py                # Flask application
├── train_model.py        # CNN model training script
├── mask_model.h5         # Trained model (Git LFS)
├── requirements.txt
└── README.md

## Model Architecture

- Convolution and MaxPooling layers
- Flatten layer
- Dropout for regularization
- Dense layer with ReLU activation
- Output layer with Softmax activation (3 classes)

## Data Augmentation

Training data is augmented using rotation, shifting, zooming, shearing, flipping, and rescaling.

## How to Run the Project

1. Clone the repository
git clone https://github.com/csngt/mask-detection-system.git
cd mask-detection-system

2. Install dependencies
pip install -r requirements.txt

3. Run the application
python app.py

4. Open browser and go to
http://127.0.0.1:5000/

## Model Training (Optional)

To retrain the model:
python train_model.py

This will generate a new mask_model.h5 file.

## Notes

- The trained model file is stored using Git LFS.
- The dataset is not included due to size limitations.
- Haar Cascade is used for face detection.

## Author

Anandu Murali
B.Tech Computer Science Graduate
