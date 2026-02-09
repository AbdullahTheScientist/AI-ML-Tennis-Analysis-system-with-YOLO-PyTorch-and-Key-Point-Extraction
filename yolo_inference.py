from ultralytics import YOLO

model = YOLO("models/yolo5_last.pt")  # load a pretrained YOLOv8n model

print(model.names)  # get class names
results = model.predict(source="input_videos/input_video.mp4", save=True, conf=0.5)  # predict on an image or video and display the results


# print(results[0].boxes)
for box in results[0].boxes:
    print(box)