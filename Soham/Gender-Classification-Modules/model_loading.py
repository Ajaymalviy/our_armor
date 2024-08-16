import torch
from tensorflow.keras.models import load_model

# Singeton model loader
class ModelLoader:
    _instance = None

    @staticmethod
    def get_instance():
        if ModelLoader._instance is None:
            ModelLoader._instance = ModelLoader()
        return ModelLoader._instance

    def __init__(self):
        if ModelLoader._instance is not None:
            raise Exception("This class is a singleton!")
        self.gender_model = load_model('Models/gender_classification_model_mobilenetv2.h5')
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def get_models(self):
        return self.gender_model, self.yolo_model
