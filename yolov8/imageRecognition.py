from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # initialize model
results = model("./images/dog1.jpeg")  # perform inference
results[0].show()  # display results for the first image
