# app.py — Traffic Analytics (Stable Streamlit Version)

import os
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
# Utility
# -------------------------------------------------------

def bb_centroid(bb):
    x1, y1, x2, y2 = bb
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def build_homography(px_points, rect_w, rect_l):
    src = np.array(px_points, dtype=np.float32)
    dst = np.array(
        [[0, 0], [rect_w, 0], [rect_w, rect_l], [0, rect_l]],
        dtype=np.float32
    )
    return cv2.getPerspectiveTransform(src, dst)

def project_point(H, cx, cy):
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    gp = cv2.perspectiveTransform(pt, H)[0][0]
    return float(gp[0]), float(gp[1])

def side_of_line(px, py, x1, y1, x2, y2):
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

# -------------------------------------------------------
# Tracker (FIXED SAFE VERSION)
# -------------------------------------------------------

class Tracker:
    def __init__(self, fps, H=None):
        self.fps = fps
        self.H = H
        self.tracks = {}
        self.next_id = 1

    def update(self, dets, frame_idx):

        for d in dets:

            if "cx" not in d or "cy" not in d:
                continue

            matched = False

            for tid, tr in self.tracks.items():

                if "cx" not in tr or "cy" not in tr:
                    continue

                dist = np.hypot(d["cx"] - tr["cx"], d["cy"] - tr["cy"])

                if dist < 60:
                    tr["bbox"] = d["bbox"]
                    tr["prev"] = tr["curr"]
                    tr["curr"] = (d["cx"], d["cy"])
                    tr["cx"] = d["cx"]
                    tr["cy"] = d["cy"]
                    tr["last"] = frame_idx

                    if self.H is not None:
                        gx, gy = project_point(self.H, d["cx"], d["cy"])
                        tr["hist"].append((frame_idx, gx, gy))

                    d["id"] = tid
                    matched = True
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
                    "cx": d["cx"],
                    "cy": d["cy"],
                    "hist": hist,
                    "mph": None,
                    "last": frame_idx,
                    "crossed": False
                }

                d["id"] = tid

        # remove stale
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

def analyze(video_path, H, line_y_frac):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolov8n.pt")
    tracker = Tracker(fps, H)

    out_path = "annotated.mp4"
    writer = make_writer(out_path, fps, w, h)

    line_y = int(line_y_frac * h)
    count = 0
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

        # SPEED
        for tr in tracker.tracks.values():
            if H is not None and len(tr["hist"]) >= 2:
                f0, x0, y0 = tr["hist"][-2]
                f1, x1, y1 = tr["hist"][-1]
                dt = (f1 - f0) / fps
                if dt > 0:
                    dist = np.hypot(x1 - x0, y1 - y0)
                    tr["mph"] = (dist / dt) * 2.23694

        # DRAW LINE
        cv2.line(frame, (0, line_y), (w, line_y), CYAN, 3)

        # DRAW TRACKS + COUNT
        for tid, tr in tracker.tracks.items():

            x1, y1, x2, y2 = map(int, tr["bbox"])

            crossed = False

            if not tr["crossed"]:
                if tr["prev"][1] < line_y and tr["curr"][1] >= line_y:
                    tr["crossed"] = True
                    crossed = True
                    count += 1

                    events.append({
                        "time_s": round(frame_idx / fps, 2),
                        "car_id": tid,
                        "mph": round(tr["mph"], 1) if tr["mph"] else None
                    })

            color = GREEN if tr["crossed"] else YELLOW

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            mph_text = (
                f"{tr['mph']:.1f} mph"
                if tr["mph"] is not None
                else "—"
            )

            cv2.putText(
                frame,
                f"ID {tid} | {mph_text}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.putText(
            frame,
            f"COUNT: {count}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            WHITE,
            2
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(events)

    return out_path, df, count

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

st.title("Traffic Analytics")

uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

use_calib = st.checkbox("Enable Speed Calibration")

line_y_frac = st.slider("Count line position", 0.1, 0.9, 0.5)

if use_calib:
    st.subheader("Calibration Rectangle (4 Points)")
    rect_w = st.number_input("Width (m)", 0.1, 10.0, 3.7)
    rect_l = st.number_input("Length (m)", 1.0, 30.0, 10.0)

    p1x = st.number_input("P1 x", 0, 2000, 100)
    p1y = st.number_input("P1 y", 0, 2000, 100)
    p2x = st.number_input("P2 x", 0, 2000, 300)
    p2y = st.number_input("P2 y", 0, 2000, 100)
    p3x = st.number_input("P3 x", 0, 2000, 300)
    p3y = st.number_input("P3 y", 0, 2000, 300)
    p4x = st.number_input("P4 x", 0, 2000, 100)
    p4y = st.number_input("P4 y", 0, 2000, 300)

run = st.button("Run analysis")

if run and uploaded:

    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded.name)

    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    H = None
    if use_calib:
        px_points = [
            (p1x, p1y),
            (p2x, p2y),
            (p3x, p3y),
            (p4x, p4y)
        ]
        H = build_homography(px_points, rect_w, rect_l)

    with st.spinner("Processing..."):
        out_path, events_df, total = analyze(
            video_path,
            H,
            line_y_frac
        )

    st.success("Done")

    st.video(out_path)
    st.write("Total crossings:", total)

    if not events_df.empty:
        st.dataframe(events_df)
