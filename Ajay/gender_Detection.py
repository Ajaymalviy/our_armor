import cv2
import numpy as np
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file


depth = 16
k = 8
weight_file = get_file("weights.28-3.73.hdf5", 
                       "https://github.com/yu4u/age-gender-estimation/releases/download/0.5/weights.28-3.73.hdf5", 
                       cache_subdir="pretrained_models",
                       file_hash="3e3e9658fa3779958d2f3baccd157ba0")

model = WideResNet(64, depth=depth, k=k)()
model.load_weights(weight_file)

# Function to extract frames from video
def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def preprocess_faces(frames, faces, size=(64, 64)):
    processed_faces = []
    for i, face_list in enumerate(faces):
        for face in face_list:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_img = frames[i][y:y+h, x:x+w]
            face_img = cv2.resize(face_img, size)
            processed_faces.append(face_img)
    return np.array(processed_faces)


def detect_and_predict_gender(video_path):
    frames = extract_frames(video_path)
    faces = detect_faces(frames)
    processed_faces = preprocess_faces(frames, faces)
    
    results = model.predict(processed_faces)
    genders = ['Male' if pred[0] > 0.5 else 'Female' for pred in results[1]]
    
    return genders


video_path = 'path_to_your_video.mp4'
genders = detect_and_predict_gender(video_path)

print(genders)
