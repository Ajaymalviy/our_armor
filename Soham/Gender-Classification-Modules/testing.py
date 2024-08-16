import torch
from PIL import Image

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load an image
img = Image.open('Images/lol.jpg')

# Perform inference
results = model(img)

# Print results
results.show()  # or results.save() to save the image with detections
