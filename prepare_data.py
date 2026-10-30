import os
import cv2
import csv
import time
from ultralytics import YOLO

# --- STEP 1: Resize the original video ---
model = YOLO('yolo11n.pt')  # Load YOLO model

input_path = 'people.mp4'
output_path = 'people_resized.mp4'
target_width, target_height = 640, 360

cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (target_width, target_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (target_width, target_height))
    out.write(resized_frame)

cap.release()
out.release()
print("Video resized and saved as:", output_path)

# --- STEP 2: Run YOLO detection and save results ---

video_file = 'people_resized.mp4'
cap = cv2.VideoCapture(video_file)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_detected.mp4', fourcc, fps, (width, height))

csv_file = open('detections.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['frame', 'x1', 'y1', 'x2', 'y2', 'confidence'])

frame_id = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.2, classes=[0], verbose=False)[0]

    for box in results.boxes:
        conf = box.conf[0].item()
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        writer.writerow([frame_id, x1, y1, x2, y2, conf])

        if conf >= 0.5:
            color = (0, 255, 0)
        elif conf >= 0.1:
            color = (0, 0, 255)
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)
    frame_id += 1

end_time = time.time()
cap.release()
out.release()
csv_file.close()

total_time = end_time - start_time
estimated_fps = frame_id / total_time

print("Done!")
print(f"Processed {frame_id} frames in {total_time:.2f} seconds")
print(f"Estimated FPS: {estimated_fps:.2f}")
