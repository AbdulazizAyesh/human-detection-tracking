import cv2
import numpy as np
import pandas as pd

def generate_heatmap_video(tracking_csv='tracked_output.csv', input_video='people_resized.mp4', output_video='heatmap_video.mp4'):
    df = pd.read_csv(tracking_csv)
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    heatmap = np.zeros((height, width), dtype=np.float32)
    frame_map = df.groupby("frame")

    frame_id = 0
    while frame_id < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in frame_map.groups:
            for _, row in frame_map.get_group(frame_id).iterrows():
                x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
                heatmap[y1:y2, x1:x2] += 1

        normalized = np.clip(heatmap / heatmap.max() * 255, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, colored_heatmap, 0.4, 0)
        out.write(overlay)
        frame_id += 1

    cap.release()
    out.release()
    print("Heatmap video saved as:", output_video)
