import os
import time
import tempfile
from itertools import combinations
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
CAR_CLASS_ID = 2
NEAR_MISS_THRESHOLD_S = 4.0
MAX_TRACK_AGE = 20

# =========================
# UTILS
# =========================
def bb_centroid(bb):
    x1, y1, x2, y2 = bb
    return (x1 + x2) / 2, (y1 + y2) / 2

def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def build_homography(px_pts, width_m, length_m):
    src = np.array(px_pts, dtype=np.float32)
    dst = np.array([
        [0, 0],
        [width_m, 0],
        [width_m, length_m],
        [0, length_m]
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)

def project_point(H, x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0][0]
    return float(out[0]), float(out[1])

# =========================
# TRACKER
# =========================
class Tracker:
    def __init__(self, fps, H=None):
        self.fps = fps
        self.H = H
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_idx):
        updated = {}

        for det in detections:
            matched_id = None
            cx, cy = det["cx"], det["cy"]

            for tid, tr in self.tracks.items():
                dist = np.hypot(cx - tr["cx"], cy - tr["cy"])
                if dist < 60:
                    matched_id = tid
                    break

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
                updated[matched_id] = {
                    "id": matched_id,
                    "cx": cx,
                    "cy": cy,
                    "bbox": det["bbox"],
                    "last_frame": frame_idx,
                    "hist": deque(maxlen=120),
                }
            else:
                tr = self.tracks[matched_id]
                tr["cx"], tr["cy"] = cx, cy
                tr["bbox"] = det["bbox"]
                tr["last_frame"] = frame_idx
                updated[matched_id] = tr

            tr = updated[matched_id]
            if self.H is not None:
                gx, gy = project_point(self.H, cx, cy)
                tr["hist"].append((frame_idx, gx, gy))

        self.tracks = {
            tid: tr for tid, tr in updated.items()
            if frame_idx - tr["last_frame"] <= MAX_TRACK_AGE
        }

        return self.tracks

# =========================
# NEAR MISS DETECTOR
# =========================
class NearMissDetector:
    def __init__(self, zone):
        self.zone = zone
        self.events = []

    def in_zone(self, x, y):
        return (
            self.zone["xmin"] <= x <= self.zone["xmax"]
            and self.zone["ymin"] <= y <= self.zone["ymax"]
        )

    def check(self, tracks, t_s):
        active = []

        for tr in tracks.values():
            if len(tr["hist"]) < 2:
                continue
            _, x, y = tr["hist"][-1]
            if self.in_zone(x, y):
                active.append(tr)

        for a, b in combinations(active, 2):
            ta = a["hist"][-1][0]
            tb = b["hist"][-1][0]
            pet = abs(ta - tb) / 30.0

            if pet <= NEAR_MISS_THRESHOLD_S:
                self.events.append({
                    "time_s": round(t_s, 2),
                    "car_1": a["id"],
                    "car_2": b["id"],
                    "pet_s": round(pet, 2)
                })

# =========================
# STREAMLIT UI
# =========================
st.title("Vehicle Safety Event Detection")

uploaded = st.file_uploader("Upload video", type=["mp4", "mov"])

model = YOLO("yolov8n.pt")

use_calib = st.checkbox("Enable speed calibration", True)

if use_calib:
    st.subheader("Homography")
    rect_w = st.number_input("Width meters", 3.7)
    rect_l = st.number_input("Length meters", 10.0)
    pts = []
    for i in range(4):
        x = st.number_input(f"P{i+1} x", 0)
        y = st.number_input(f"P{i+1} y", 0)
        pts.append((x, y))

st.subheader("Conflict Zone meters")
zx1 = st.number_input("xmin", 2.0)
zx2 = st.number_input("xmax", 6.0)
zy1 = st.number_input("ymin", 4.0)
zy2 = st.number_input("ymax", 8.0)

run = st.button("Run")

# =========================
# MAIN
# =========================
if run and uploaded:
    tmp = tempfile.mkdtemp()
    video_path = os.path.join(tmp, uploaded.name)
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(tmp, "annotated.mp4")
    writer = make_writer(out_path, fps, w, h)

    H = None
    if use_calib:
        H = build_homography(pts, rect_w, rect_l)

    tracker = Tracker(fps, H)
    near_miss = NearMissDetector({
        "xmin": zx1,
        "xmax": zx2,
        "ymin": zy1,
        "ymax": zy2,
    })

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, conf=0.25, verbose=False)[0]

        dets = []
        if res.boxes is not None:
            for bb, cls in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int),
            ):
                if cls == CAR_CLASS_ID:
                    cx, cy = bb_centroid(bb)
                    dets.append({"cx": cx, "cy": cy, "bbox": bb})

        tracks = tracker.update(dets, frame_idx)
        t_s = frame_idx / fps
        near_miss.check(tracks, t_s)

        # draw zone
        if H:
            p1 = (int(zx1 * 50), int(zy1 * 50))
            p2 = (int(zx2 * 50), int(zy2 * 50))
            cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)

        for tr in tracks.values():
            x1, y1, x2, y2 = map(int, tr["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {tr['id']}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(near_miss.events)

    st.success("Done")
    st.video(out_path)
    st.dataframe(df)

    csv_path = os.path.join(tmp, "near_miss_events.csv")
    df.to_csv(csv_path, index=False)
    st.download_button("Download CSV", open(csv_path, "rb"), "near_miss_events.csv")
