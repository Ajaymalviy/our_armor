import cv2
import numpy as np
from collections import deque
from iou_calculator import calculate_iou
from preprocessing import preprocess_for_classification

class PersonTracker:
    def __init__(self, yolo_model, gender_model):
        self.yolo_model = yolo_model
        self.gender_model = gender_model
        self.frame_count = 0
        self.tracking_data = {}
        self.next_person_id = 0

    def process_frame(self, frame):
        self.frame_count += 1

        if self.frame_count % 5 == 0:
            self.detect_persons(frame)

        self.update_tracking(frame)
        self.cleanup_tracking()

    def detect_persons(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(img_rgb)
        new_detections = []

        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            if cls == 0 and conf > 0.5:
                x, y, w, h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                new_detections.append([x, y, w, h, conf])

        self.match_detections(new_detections)

    def match_detections(self, new_detections):
        updated_tracking_data = {}
        for detection in new_detections:
            x, y, w, h, confidence = detection
            matched = False

            for person_id, data in self.tracking_data.items():
                tracked_box = data['box']
                iou = calculate_iou(tracked_box, (x, y, w, h))

                if iou > 0.5:
                    data['box'] = (x, y, w, h)
                    data['frames_seen'] = self.frame_count
                    updated_tracking_data[person_id] = data
                    matched = True
                    break

            if not matched:
                updated_tracking_data[self.next_person_id] = {
                    'box': (x, y, w, h),
                    'gender_votes': deque(maxlen=15),
                    'fixed_gender': None,
                    'frames_seen': self.frame_count
                }
                self.next_person_id += 1

        self.tracking_data = updated_tracking_data

    def update_tracking(self, frame):
        for person_id, data in list(self.tracking_data.items()):
            x, y, w, h = data['box']
            person_img = frame[y:y + h, x:x + w]

            if data['fixed_gender'] is None:
                self.classify_gender(data, person_img)

            gender = data['fixed_gender']
            color = (0, 255, 0) if gender == 'Male' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{gender} ({self.frame_count - data["frames_seen"]} frames)', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def classify_gender(self, data, person_img):
        input_img = preprocess_for_classification(person_img)
        gender_pred = self.gender_model.predict(input_img)[0][0]
        gender = 'Male' if gender_pred < 0.5 else 'Female'
        data['gender_votes'].append(gender)

        if len(data['gender_votes']) >= data['gender_votes'].maxlen:
            vote_count = np.array(data['gender_votes'])
            male_votes = np.sum(vote_count == 'Male')
            female_votes = np.sum(vote_count == 'Female')

            if male_votes > female_votes:
                data['fixed_gender'] = 'Male'
            elif female_votes > male_votes:
                data['fixed_gender'] = 'Female'
            else:
                data['fixed_gender'] = 'Uncertain'

    def cleanup_tracking(self):
        for person_id, data in list(self.tracking_data.items()):
            if self.frame_count - data['frames_seen'] > 150:
                del self.tracking_data[person_id]
