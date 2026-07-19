from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

input_folder = r"D:\Retail Theft Detection\dataset\MOT17\train\MOT17-02-FRCNN\img1"

output_folder = r"D:\Retail Theft Detection\outputs\detection"

os.makedirs(output_folder, exist_ok=True)

frames = sorted(os.listdir(input_folder))[:100]

for frame_name in frames:

    frame_path = os.path.join(input_folder, frame_name)

    frame = cv2.imread(frame_path)

    results = model(frame, classes=[0])

    annotated = results[0].plot()

    output_path = os.path.join(output_folder, frame_name)

    cv2.imwrite(output_path, annotated)

print("Detection complete.")