import sys
import os
import time
from flask import Flask, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
from camera import VideoCamera
from recognize1 import FaceRecognizer
from add_persons import add_persons
from add.helper import Helper

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

app = Flask(__name__, template_folder='templates')

recognizer = FaceRecognizer(config_path="./face_tracking/config/config_tracking.yaml")
camera = VideoCamera(source="video/6071425745590.mp4")

BACKUP_DIR = "./datasets/backup"
ADD_PERSONS_DIR = "./datasets/new_persons"
FACES_SAVE_DIR = "./datasets/data"
FEATURES_PATH = "./datasets/face_features/feature"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

classifier = Helper()

def load_model():
    pass

def draw_centroid(frame, bounding_boxes):
    for (x, y, w, h) in bounding_boxes:
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    return frame

def draw_pose_landmarks(frame, landmarks):
    if landmarks:
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    return frame

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

        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            processed_frame = draw_pose_landmarks(processed_frame, results.pose_landmarks.landmark)

        bounding_boxes = [(50, 50, 100, 200), (200, 150, 100, 200)]
        processed_frame = draw_centroid(processed_frame, bounding_boxes)

        classification = classifier.classify_frame()
        cv2.putText(processed_frame, f"Classification: {classification}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
    except Exception as e:
        pass
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
