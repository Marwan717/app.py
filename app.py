# app.py — Traffic Analytics + Calibrated MPH + Forward Count
# Streamlit Cloud Safe Version

import os
import json
import time
import tempfile
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


CAR_CLASS_ID = 2

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


def bb_centroid(bb):
    x1, y1, x2, y2 = bb
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def build_homography(px_points, rect_w, rect_l):
    src = np.array(px_points, dtype=np.float32)
    dst = np.array(
        [[0, 0], [rect_w, 0], [rect_w, rect_l], [0, rect_l]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def project_point(H, cx, cy):
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    gp = cv2.perspectiveTransform(pt, H)[0][0]
    return float(gp[0]), float(gp[1])


# -------------------------------------------------------
# Tracker
# -------------------------------------------------------

class Tracker:
    def __init__(self, fps, H=None):
        self.fps = fps
        self.H = H
        self.tracks = {}
        self.next_id = 1

    def update(self, dets, frame_idx):
        updated_ids = set()

        for d in dets:
            matched = False

            for tid, tr in self.tracks.items():
                dist = np.hypot(d["cx"] - tr["cx"], d["cy"] - tr["cy"])
                if dist < 60:
                    tr["bbox"] = d["bbox"]
                    tr["prev"] = tr["curr"]
                    tr["curr"] = (d["cx"], d["cy"])
                    tr["last"] = frame_idx

                    if self.H is not None:
                        gx, gy = project_point(self.H, d["cx"], d["cy"])
                        tr["hist"].append((frame_idx, gx, gy))

                    d["id"] = tid
                    matched = True
                    updated_ids.add(tid)
                    break

            if not matched:
                tid = self.next_id
                self.next_id += 1

                hist = deque(maxlen=60)
                if self.H is not None:
                    gx, gy = project_point(self.H, d["cx"], d["cy"])
                    hist.append((frame_idx, gx, gy))

                self.tracks[tid] = {
                    "bbox": d["bbox"],
                    "curr": (d["cx"], d["cy"]),
                    "prev": (d["cx"], d["cy"]),
                    "hist": hist,
                    "mph": None,
                    "last": frame_idx,
                    "crossed": False,
                }

                d["id"] = tid
                updated_ids.add(tid)

        # Remove stale tracks
        dead = [
            tid for tid, tr in self.tracks.items()
            if frame_idx - tr["last"] > 30
        ]
        for tid in dead:
            del self.tracks[tid]

        return dets


# -------------------------------------------------------
# Analysis
# -------------------------------------------------------

def analyze(video_path, H, enable_speed, line_y_frac):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    line_y = int(h * line_y_frac)

    model = YOLO("yolov8n.pt")
    tracker = Tracker(fps, H)

    temp_dir = tempfile.mkdtemp()
    out_path = os.path.join(temp_dir, "output.mp4")
    writer = make_writer(out_path, fps, w, h)

    forward_count = 0
    events = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        dets = []

        if results.boxes is not None:
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                if int(cls) != CAR_CLASS_ID:
                    continue

                bb = box.cpu().numpy()
                cx, cy = bb_centroid(bb)

                dets.append({
                    "bbox": bb,
                    "cx": cx,
                    "cy": cy
                })

        dets = tracker.update(dets, frame_idx)

        cv2.line(frame, (0, line_y), (w, line_y), CYAN, 3)

        for tid, tr in tracker.tracks.items():

            # Speed
            if enable_speed and H is not None and len(tr["hist"]) >= 2:
                f0, x0, y0 = tr["hist"][-2]
                f1, x1, y1 = tr["hist"][-1]
                dt = (f1 - f0) / fps
                if dt > 0:
                    dist = np.hypot(x1 - x0, y1 - y0)
                    tr["mph"] = (dist / dt) * 2.23694

            # Crossing
            prev_y = tr["prev"][1]
            curr_y = tr["curr"][1]

            if prev_y < line_y <= curr_y and not tr["crossed"]:
                tr["crossed"] = True
                forward_count += 1

                events.append({
                    "time_s": round(frame_idx / fps, 2),
                    "car_id": tid,
                    "mph": round(tr["mph"], 1) if tr["mph"] else None
                })

            color = GREEN if tr["crossed"] else YELLOW

            x1, y1, x2, y2 = map(int, tr["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            mph_text = (
                f"{tr['mph']:.1f} mph"
                if tr["mph"] is not None
                else "—"
            )

            cv2.putText(
                frame,
                f"ID {tid} {mph_text}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.putText(
            frame,
            f"COUNT: {forward_count}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            WHITE,
            2,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    events_df = pd.DataFrame(events)

    return out_path, events_df, forward_count


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

st.title("Traffic Analytics")

uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

st.sidebar.header("Settings")

enable_speed = st.sidebar.checkbox("Enable Speed Calibration", True)
line_y_frac = st.sidebar.slider("Line Position", 0.2, 0.9, 0.55, 0.01)

if enable_speed:
    rect_w = st.sidebar.number_input("Road width (m)", value=3.7)
    rect_l = st.sidebar.number_input("Road length (m)", value=10.0)

    st.sidebar.caption("4 calibration points (clockwise)")
    p1x = st.sidebar.number_input("P1 x", value=100)
    p1y = st.sidebar.number_input("P1 y", value=100)
    p2x = st.sidebar.number_input("P2 x", value=300)
    p2y = st.sidebar.number_input("P2 y", value=100)
    p3x = st.sidebar.number_input("P3 x", value=300)
    p3y = st.sidebar.number_input("P3 y", value=300)
    p4x = st.sidebar.number_input("P4 x", value=100)
    p4y = st.sidebar.number_input("P4 y", value=300)

run = st.button("Run analysis", disabled=uploaded is None)

if run and uploaded is not None:

    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded.name)

    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    H = None
    if enable_speed:
        H = build_homography(
            [(p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)],
            rect_w,
            rect_l,
        )

    with st.spinner("Processing..."):
        out_path, events_df, count = analyze(
            video_path,
            H,
            enable_speed,
            line_y_frac,
        )

    st.success("Done")

    st.metric("Forward crossings", count)

    st.video(out_path)

    st.subheader("Crossing Events")
    st.dataframe(events_df)
