import cv2
from queue import Queue
from model_loading import ModelLoader
from frame_capture import FrameCapture
from person_tracker import PersonTracker

def main(video_path):
    # Load models through the singleton instance
    model_loader = ModelLoader.get_instance()
    gender_model, yolo_model = model_loader.get_models()

    # Initialize the frame capture thread
    frame_queue = Queue()
    capture_thread = FrameCapture(video_path, frame_queue)
    capture_thread.start()

    # Initialize the person tracker with the loaded models
    person_tracker = PersonTracker(yolo_model, gender_model)

    # Process frames from the video
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            person_tracker.process_frame(frame)
            cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    capture_thread.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 0 # Replace with 0 for webcam
    main(video_path)
