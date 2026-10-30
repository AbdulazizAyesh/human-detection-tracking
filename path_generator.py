import cv2
import pandas as pd
import numpy as np

def generate_path_video(tracking_csv='tracked_output.csv',
                        input_video='people_resized.mp4',
                        output_video='path_video.mp4',
                        selected_id=None):
    if selected_id is None or selected_id == 'All':
        return  # Don't generate path video for 'All'

    df = pd.read_csv(tracking_csv)
    person_df = df[df['id'] == selected_id]

    if person_df.empty:
        print(f"No data found for ID {selected_id}")
        return

    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_map = person_df.groupby("frame")
    path_points = []

    frame_id = 0
    while frame_id < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in frame_map.groups:
            row = frame_map.get_group(frame_id).iloc[0]
            x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            path_points.append(center)

        for i in range(1, len(path_points)):
            cv2.line(frame, path_points[i - 1], path_points[i], (0, 0, 255), 2)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    print(f"Path video saved as: {output_video}")
