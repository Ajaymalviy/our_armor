import cv2
from threading import Thread

class FrameCapture(Thread):
    def __init__(self, src, queue):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        self.queue = queue
        self.stopped = False

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            if self.queue.qsize() < 10:
                self.queue.put(frame)

    def stop(self):
        self.stopped = True
        self.cap.release()
