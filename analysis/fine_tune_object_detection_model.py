from ultralytics import YOLO

# Load a COCO-pretrained YOLO12n model
model = YOLO("yolo12n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="./dataset/Senior-Project/data.yaml", epochs=100, imgsz=640)
results = model.train(data="/Users/yunhocho/Documents/GitHub/RoboTech-Hackathon-2025/analysis/dataset/data.yaml", epochs=100, imgsz=640)
