from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # Load a pretrained YOLOv8 model

# Train the model
model.train(data='data.yaml', epochs=15, imgsz=640, batch=16 , project='runs/train', name='exp')
