import numpy as np
import cv2
import imageio

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

def detect_people_in_region(frame, region, net, output_layers, classes):
    x, y, w, h = region
    cropped_frame = frame[y:y+h, x:x+w]
    boxes, indexes, class_ids = detect_people(cropped_frame, net, output_layers, classes)        
    region_box = np.array([[0, 0, w, h]]) 
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

    print(f"Number of people detected in the specified region: {num_people}")

    if num_people > 2:
        print(f"Alert in the specified region: More than 2 people detected!")
    frame[y:y+h, x:x+w] = cropped_frame
    return frame

def create_regions(frame_width, frame_height, num_regions):
    if num_regions % 4 != 0:
        raise ValueError("Number of regions must be a multiple of 4")
    
    grid_size = int(np.sqrt(num_regions))  # Ensure square grid size
    region_width = frame_width // grid_size
    region_height = frame_height // grid_size

    regions = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = j * region_width
            y = i * region_height
            regions.append((x, y, region_width, region_height))
    return regions

def main(video_source, num_regions, region_indices):
    cv2.startWindowThread()
    net, output_layers, classes = load_yolo_model()

    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError("Could not open video source with OpenCV")
    except:
        cap = imageio.get_reader(video_source, 'ffmpeg')

    out = cv2.VideoWriter(
        'detected_person.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        15.,
        (640, 480))

    while True:
        ret, frame = None, None
        if isinstance(cap, cv2.VideoCapture):
            ret, frame = cap.read()
        else:
            try:
                frame = cap.get_next_data()
                ret = True
            except IndexError:
                ret = False

        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        height, width, _ = frame.shape
        regions = create_regions(width, height, num_regions)

        # Draw all regions
        for region in regions:
            x, y, w, h = region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    
        for region_index in region_indices:
            if 0 <= region_index < len(regions):
                frame = detect_people_in_region(frame, regions[region_index], net, output_layers, classes)
            else:
                print(f"Invalid region index provided: {region_index}")

        out.write(frame.astype('uint8'))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if isinstance(cap, cv2.VideoCapture):
        cap.release()
    out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    video_source = "realtime.mp4"   #"http://192.168.1.4:4747/video" "
    num_regions = 12 
    region_indices = [4]  
    main(video_source, num_regions, region_indices)
