import os
import pandas as pd
import streamlit as st
import cv2
import time

from tracking import generate_tracking
from heatmap_generator import generate_heatmap_video
from path_generator import generate_path_video

# Auto-generate tracking and heatmap data
if not os.path.exists('tracked_output.csv'):
    generate_tracking()
if not os.path.exists('heatmap_video.mp4'):
    generate_heatmap_video()

# Load tracking data
df = pd.read_csv('tracked_output.csv')

# Session state defaults
if 'frame_id' not in st.session_state:
    st.session_state.frame_id = 0
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'speed_multiplier' not in st.session_state:
    st.session_state.speed_multiplier = 1.0
if 'mode' not in st.session_state:
    st.session_state.mode = 'video'
if 'last_selected_id' not in st.session_state:
    st.session_state.last_selected_id = None

# UI Setup
st.title("People Detection and Tracking")

all_ids = sorted(df['id'].unique())
selected_id = st.selectbox("Select a Person ID", ['All'] + all_ids)

# If ID changed, generate new path video
if selected_id != 'All' and selected_id != st.session_state.last_selected_id:
    generate_path_video(selected_id=selected_id)
    st.session_state.last_selected_id = selected_id

# Duration display
fps = 30
durations = df.groupby('id')['frame'].agg(lambda f: round((f.max() - f.min() + 1) / fps, 1))
if selected_id != 'All' and selected_id in durations:
    st.markdown(f"Person {selected_id} appeared for `{durations[selected_id]} seconds`")

# Mode switch
switch_label = "Switch to Slider Mode" if st.session_state.mode == "video" else "Switch to Video Mode"
if st.button(switch_label):
    if st.session_state.mode == "video":
        st.session_state.mode = "slider"
        st.session_state.playing = False
    else:
        st.session_state.mode = "video"
        st.session_state.playing = True

# Load videos
video_path = 'people_resized.mp4'
heatmap_path = 'heatmap_video.mp4'
path_video_path = 'path_video.mp4'

cap = cv2.VideoCapture(video_path)
heat_cap = cv2.VideoCapture(heatmap_path)
path_cap = cv2.VideoCapture(path_video_path) if selected_id != 'All' else None

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

# Speed or slider
if st.session_state.mode == "video":
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("0.5x"):
            st.session_state.speed_multiplier = 0.5
            st.session_state.playing = True
    with col2:
        if st.button("1x"):
            st.session_state.speed_multiplier = 1.0
            st.session_state.playing = True
    with col3:
        if st.button("2x"):
            st.session_state.speed_multiplier = 2.0
            st.session_state.playing = True
else:
    slider_val = st.slider("Seek Frame", 0, total_frames - 1, st.session_state.frame_id)
    if slider_val != st.session_state.frame_id:
        st.session_state.frame_id = slider_val
        st.session_state.playing = False

# Frame display
frame_display = st.empty()
heat_display = st.empty()
path_display = st.empty()
frame_map = df.groupby('frame')

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_id)
    heat_cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_id)
    if path_cap:
        path_cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_id)

    ret, frame = cap.read()
    ret_h, heat_frame = heat_cap.read()
    ret_p, path_frame = (path_cap.read() if path_cap else (False, None))

    if not ret or not ret_h or (path_cap and not ret_p):
        st.session_state.playing = False
        break

    # Draw boxes
    if st.session_state.frame_id in frame_map.groups:
        frame_dets = frame_map.get_group(st.session_state.frame_id)
        if selected_id != 'All':
            frame_dets = frame_dets[frame_dets['id'] == selected_id]
        for _, row in frame_dets.iterrows():
            x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
            pid = row['id']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f'ID {pid}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Tracked Video")
    if selected_id != 'All' and path_frame is not None:
        col1, col2 = st.columns(2)
        with col1:
            heat_display.image(cv2.cvtColor(heat_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Heatmap Video")
        with col2:
            path_display.image(cv2.cvtColor(path_frame, cv2.COLOR_BGR2RGB), channels="RGB",
                               caption=f"Path of {selected_id}")
    else:
        heat_display.image(cv2.cvtColor(heat_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Heatmap Video")

    if st.session_state.mode == "slider" or not st.session_state.playing:
        break

    time.sleep((1.0 / actual_fps) / st.session_state.speed_multiplier)
    st.session_state.frame_id += 1

    if st.session_state.frame_id >= total_frames:
        st.session_state.playing = False
        break

cap.release()
heat_cap.release()
if path_cap:
    path_cap.release()
