import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_for_classification(image):
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image
