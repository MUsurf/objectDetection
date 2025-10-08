print('hi')
from ultralytics import YOLO

#load a model
model = YOLO('yolov8n.yaml')
#use the model
results = model.train(data = "cat.yaml" , epochs = 3) #train the model
