import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Model and Face Detector
model = load_model('mask_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels_dict = {0: 'Incorrectly Worn', 1: 'With Mask', 2: 'Without Mask'}
colors_dict = {0: (0, 255, 255), 1: (0, 255, 0), 2: (0, 0, 255)}

def process_and_predict(image_path):
    """Helper function to process an image and return prediction details."""
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    idx = np.argmax(prediction)
    return labels_dict[idx], round(np.max(prediction) * 100, 2)

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success: 
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # 1. Preprocessing
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (150, 150)) / 255.0
            face_img = np.expand_dims(img_to_array(face_img), axis=0)
            
            # 2. Prediction & Confidence Calculation
            prediction = model.predict(face_img, verbose=0)
            idx = np.argmax(prediction)
            confidence = prediction[0][idx] * 100  # Convert to percentage
            
            # 3. UI: Draw Face Box and Label
            color = colors_dict[idx]
            label = f"{labels_dict[idx]}: {confidence:.1f}%"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 4. UI: Draw the Confidence Bar
            # Background bar (Dark Gray)
            cv2.rectangle(frame, (x, y+h+10), (x+w, y+h+25), (50, 50, 50), -1)
            
            # Dynamic foreground bar (Calculated width based on confidence)
            bar_width = int((confidence / 100) * w)
            cv2.rectangle(frame, (x, y+h+10), (x+bar_width, y+h+25), color, -1)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return "No file uploaded"
    file = request.files['file']
    if file.filename == '': return "No file selected"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    result, confidence = process_and_predict(filepath)
    return render_template('index.html', result=result, confidence=confidence, img_path=file.filename)

if __name__ == '__main__':
    app.run(debug=True)