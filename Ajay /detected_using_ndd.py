import numpy as np
import cv2
import sys

def load_yolo_model():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes

def detect_people(frame, net, output_layers, classes):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, indexes, class_ids

def count_people_and_alert(frame, region, net, output_layers, classes):
    # Crop the frame to the region
    x, y, w, h = region
    cropped_frame = frame[y:y+h, x:x+w]

    boxes, indexes, class_ids = detect_people(cropped_frame, net, output_layers, classes)
    
    region_box = np.array([[0, 0, w, h]])  # Adjust region box for the cropped frame
    num_people = 0

    if len(indexes) > 0:
        if isinstance(indexes[0], (list, np.ndarray)):
            indexes = [i[0] for i in indexes]
        
        for i in indexes:
            bx, by, bw, bh = boxes[i]
            if (region_box[0][0] < bx < region_box[0][2] and region_box[0][1] < by < region_box[0][3]) or \
               (region_box[0][0] < bx + bw < region_box[0][2] and region_box[0][1] < by + bh < region_box[0][3]):
                num_people += 1
            label = str(classes[class_ids[i]])
            cv2.rectangle(cropped_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(cropped_frame, label, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if num_people > 2:
        print("Alert: More than 2 people detected!")
    else:
        print(f"Number of people detected: {num_people}")

    # Draw the region box on the original frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Overlay the cropped frame back onto the original frame
    frame[y:y+h, x:x+w] = cropped_frame

    return frame

def main(video_path, region):
    cv2.startWindowThread()

    net, output_layers, classes = load_yolo_model()

    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(
        'detected_person.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        15.,
        (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (600, 400))
        frame = count_people_and_alert(frame, region, net, output_layers, classes)
        
        out.write(frame.astype('uint8'))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python detect_people.py <video_path> <x> <y> <w> <h>")
        sys.exit(1)

    video_path = sys.argv[1]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    w = int(sys.argv[4])
    h = int(sys.argv[5])
    region = (x, y, w, h)

    main(video_path, region)
