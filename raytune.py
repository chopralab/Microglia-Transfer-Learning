from ultralytics import YOLO

# Load a YOLOv8n model
model = YOLO('yolov8l.pt')

# Start tuning hyperparameters for YOLOv8n training on the custom dataset
result_grid = model.tune(data='custom.yaml', use_ray=True, epochs=30)