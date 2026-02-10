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
FPS_FALLBACK = 30.0

# =========================
# UTILS
# =========================
def bb_centroid(bb):
    x1, y1, x2, y2 = bb
    return (x1 + x2) / 2, (y1 + y2) / 2

def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

# =========================
# TRACKER
# =========================
class Tracker:
    def __init__(self, fps):
        self.fps = fps
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_idx):
        updated = {}

        for det in detections:
            cx, cy = det["cx"], det["cy"]
            matched_id = None

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
            tr["hist"].append((frame_idx, cx, cy))

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
        self.logged_pairs = set()

    def in_zone(self, x, y):
        return (
            self.zone["xmin"] <= x <= self.zone["xmax"]
            and self.zone["ymin"] <= y <= self.zone["ymax"]
        )

    def check(self, tracks, t_s, fps):
        active = []

        for tr in tracks.values():
            if len(tr["hist"]) < 2:
                continue
            _, x, y = tr["hist"][-1]
            if self.in_zone(x, y):
                active.append(tr)

        for a, b in combinations(active, 2):
            pair_id = tuple(sorted([a["id"], b["id"]]))
            if pair_id in self.logged_pairs:
                continue

            fa = a["hist"][-1][0]
            fb = b["hist"][-1][0]
            pet = abs(fa - fb) / fps

            if pet <= NEAR_MISS_THRESHOLD_S:
                self.logged_pairs.add(pair_id)
                self.events.append({
                    "time_s": round(t_s, 2),
                    "car_1_id": a["id"],
                    "car_2_id": b["id"],
                    "pet_s": round(pet, 2),
                    "type": "near_miss"
                })

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("Vehicle Near-Miss Detection")

uploaded = st.file_uploader("Upload traffic video", type=["mp4", "mov", "avi"])

st.subheader("Conflict Zone (pixels)")
zx1 = st.number_input("xmin", 0)
zx2 = st.number_input("xmax", 300)
zy1 = st.number_input("ymin", 0)
zy2 = st.number_input("ymax", 300)

run = st.button("Run Analysis", type="primary")

# =========================
# MAIN
# =========================
if run and uploaded:
    tmp = tempfile.mkdtemp()
    video_path = os.path.join(tmp, uploaded.name)
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = FPS_FALLBACK

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(tmp, "annotated.mp4")
    writer = make_writer(out_path, fps, w, h)

    model = YOLO("yolov8n.pt")
    tracker = Tracker(fps)

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
        near_miss.check(tracks, t_s, fps)

        cv2.rectangle(
            frame,
            (int(zx1), int(zy1)),
            (int(zx2), int(zy2)),
            (0, 0, 255),
            2,
        )

        for tr in tracks.values():
            x1, y1, x2, y2 = map(int, tr["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {tr['id']}",
                (x1, y1 - 6),
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

    st.success("Analysis complete")
    st.video(out_path)
    st.dataframe(df)

    csv_path = os.path.join(tmp, "near_miss_events.csv")
    df.to_csv(csv_path, index=False)
    st.download_button("Download Near-Miss CSV", open(csv_path, "rb"), "near_miss_events.csv")
