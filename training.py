from ultralytics import YOLO

model = YOLO("C:\\fruitDataset\\runs\\detect\\train\\weights\\best.pt")

results = model.train(data="dataset.yaml", epochs=15, imgsz=650)

