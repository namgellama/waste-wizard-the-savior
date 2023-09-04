from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')

# Train the model
results = model.train(
    data=r'C:\Users\PREDATOR\Desktop\hackathon1\dataset', epochs=100, imgsz=64)
