import cv2
import csv
import numpy as np
import random
from sort import Sort
from collections import defaultdict

def generate_tracking():
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # Mapping from SORT ID → custom ID (e.g. H-123)
    id_mapping = {}

    def format_custom_id(sort_id):
        sort_id = int(sort_id)
        if sort_id not in id_mapping:
            random.seed(sort_id)  # Ensures consistent ID per track
            rand_digits = f"{random.randint(0, 999):03}"
            id_mapping[sort_id] = f"H-{rand_digits}"
        return id_mapping[sort_id]

    # Load detections by frame
    detections_by_frame = {}
    with open('detections.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            frame = int(row[0])
            x1, y1, x2, y2, conf = map(float, row[1:])
            detections_by_frame.setdefault(frame, []).append([x1, y1, x2, y2, conf])

    # Setup video writer
    cap = cv2.VideoCapture('people_resized.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('tracked_output.mp4', fourcc, fps, (width, height))

    # CSV writer for tracking output
    with open('tracked_output.csv', 'w', newline='') as csv_out:
        csv_writer = csv.writer(csv_out)
        csv_writer.writerow(['frame', 'id', 'x1', 'y1', 'x2', 'y2'])

        id_duration_frames = defaultdict(int)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = np.array(detections_by_frame.get(frame_id, []))
            tracks = tracker.update(detections)

            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track)
                custom_id = format_custom_id(track_id)

                id_duration_frames[custom_id] += 1
                seconds_alive = id_duration_frames[custom_id] / fps

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f'{custom_id} | {seconds_alive:.1f}s',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                csv_writer.writerow([frame_id, custom_id, x1, y1, x2, y2])

            out.write(frame)
            frame_id += 1

    cap.release()
    out.release()
    print("Tracking data generated and saved with custom IDs.")
