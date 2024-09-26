from ultralytics import YOLO
import cv2 as cv
import numpy as np

camera_index = 0
#cap = cv.VideoCapture("/Users/colejones/Library/VSProjects/Surf/Python/New York City Street Walk to Times Square. DJI Osmo Action 3.mp4")
cap = cv.VideoCapture(camera_index)
frame_width = 960
frame_height = 540
frame_rate = 30
cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv.CAP_PROP_FPS, frame_rate)

model = YOLO("yolov8n.pt")  # initialize model

if not cap.isOpened():
    print(f"Error: Camera with index {camera_index} not accessible or not found")
    exit()

## Draw a line around the orange thing

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv.imshow('Objects', annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
