import sys
import os
import time
from flask import Flask, render_template, Response, redirect, url_for
import cv2
from camera import VideoCamera
from recognize1 import FaceRecognizer
from add_persons import add_persons

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

app = Flask(__name__, template_folder='templates')

recognizer = FaceRecognizer(config_path="./face_tracking/config/config_tracking.yaml")
camera = VideoCamera(source=0)

BACKUP_DIR = "./datasets/backup"
ADD_PERSONS_DIR = "./datasets/new_persons"
FACES_SAVE_DIR = "./datasets/data"
FEATURES_PATH = "./datasets/face_features/feature"

def generate_frames():
    frame_id = 0
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        processed_frame = recognizer.process_frame(frame, frame_id, fps)

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        _, jpeg = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        frame_id += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_persons', methods=['POST'])
def add_persons_route():
    try:
        add_persons(
            backup_dir=BACKUP_DIR,
            add_persons_dir=ADD_PERSONS_DIR,
            faces_save_dir=FACES_SAVE_DIR,
            features_path=FEATURES_PATH,
        )
        print("Successfully added new persons!")
    except Exception as e:
        print(f"Error adding persons: {e}")
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
