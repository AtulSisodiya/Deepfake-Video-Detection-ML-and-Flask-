import os
import time
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import base64
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

model = load_model('fake_video_detector_model.h5')

if not os.path.exists('uploads'):
    os.makedirs('uploads')

def encode_image_to_base64(image):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized.astype("float32") / 255.0  
    return np.expand_dims(frame_normalized, axis=0) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    file = request.files['video']
    video_path = os.path.join('uploads', file.filename)
    file.save(video_path)

    socketio.emit('progress_update', {'status': 'Uploading video...', 'progress': 10})


    time.sleep(2)

    socketio.emit('progress_update', {'status': 'Checking video using AI...', 'progress': 30})
    time.sleep(2)

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()

    all_frames = [] 
    fake_frames = [] 
    real_frames = []  

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while success:
        preprocessed_frame = preprocess_frame(frame)
        prediction = model.predict(preprocessed_frame)
        result = "Fake" if prediction[0] > 0.5 else "Real"

        if result == "Fake":
            fake_frames.append(encode_image_to_base64(frame))
        else:
            real_frames.append(encode_image_to_base64(frame))

        all_frames.append(frame)
        processed_frames += 1

        progress = int((processed_frames / total_frames) * 40) + 30  
        socketio.emit('progress_update', {'status': f'Analyzing frame {processed_frames}/{total_frames}...', 'progress': progress})

        success, frame = cap.read()

    socketio.emit('progress_update', {'status': 'Video analysis complete.', 'progress': 100})

    if len(fake_frames) > 0:
        result_message = "Deepfake Video Detected"
        result_color = "red"
    else:
        result_message = "Real Video"
        result_color = "green"

    frames_to_show = fake_frames[:5]

   
    return jsonify({
        'frames': frames_to_show,
        'result_message': result_message,
        'result_color': result_color
    })

@socketio.on('connect')
def handle_connect():
    print("Client connected.")

if __name__ == '__main__':
    socketio.run(app, debug=True)
