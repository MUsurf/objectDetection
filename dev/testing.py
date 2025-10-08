from ultralytics import YOLO

model = YOLO('runs'/'weights'/'best.pt')

results = model.val