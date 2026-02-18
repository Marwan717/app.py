# app.py — Car Counting + Calibrated MPH + Near Miss Detection

import os, json, time, tempfile
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

CAR_CLASS_ID = 2

YELLOW = (0, 255, 255)
GREEN  = (0, 255, 0)
CYAN   = (255, 255, 0)
WHITE  = (255, 255, 255)
RED    = (0, 0, 255)

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def bb_centroid(bb):
    x1, y1, x2, y2 = bb
    return (x1 + x2)/2.0, (y1 + y2)/2.0

def side_of_line(px, py, x1, y1, x2, y2):
    return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)

def build_homography(px_points, rect_w, rect_l):
    src = np.array(px_points, dtype=np.float32)
    dst = np.array([[0,0],[rect_w,0],[rect_w,rect_l],[0,rect_l]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)

def project_point(H, cx, cy):
    pt = np.array([[[cx,cy]]], dtype=np.float32)
    gp = cv2.perspectiveTransform(pt, H)[0][0]
    return float(gp[0]), float(gp[1])

# ---------------------------------------------------------
# Tracker
# ---------------------------------------------------------

class CarTracker:
    def __init__(self, fps, H=None, max_age=20):
        self.fps = fps
        self.H = H
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_idx):

        # remove old tracks
        for tid in list(self.tracks.keys()):
            if frame_idx - self.tracks[tid]["last_frame"] > self.max_age:
                del self.tracks[tid]

        for det in detections:
            matched = False
            for tid, tr in self.tracks.items():
                dist = np.hypot(det["cx"]-tr["cx"], det["cy"]-tr["cy"])
                if dist < 60:
                    tr["bbox"] = det["bbox"]
                    tr["prev_cx"], tr["prev_cy"] = tr["cx"], tr["cy"]
                    tr["cx"], tr["cy"] = det["cx"], det["cy"]
                    tr["last_frame"] = frame_idx

                    if self.H is not None:
                        gx, gy = project_point(self.H, tr["cx"], tr["cy"])
                        tr["hist"].append((frame_idx,gx,gy))

                    det["track_id"] = tid
                    matched = True
                    break

            if not matched:
                tid = self.next_id
                self.next_id += 1
                tr = {
                    "id": tid,
                    "bbox": det["bbox"],
                    "cx": det["cx"],
                    "cy": det["cy"],
                    "prev_cx": det["cx"],
                    "prev_cy": det["cy"],
                    "prev_side": None,
                    "last_frame": frame_idx,
                    "crossed": False,
                    "hist": deque(maxlen=60),
                    "mph": None
                }
                if self.H is not None:
                    gx, gy = project_point(self.H, tr["cx"], tr["cy"])
                    tr["hist"].append((frame_idx,gx,gy))
                self.tracks[tid] = tr
                det["track_id"] = tid

        return detections

# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------

def analyze(video_path, H, near_dist_thresh, ttc_thresh):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolov8n.pt")
    tracker = CarTracker(fps=fps, H=H)

    output_path = "annotated_output.mp4"
    writer = make_writer(output_path, fps, w, h)

    forward_count = 0
    near_miss_count = 0
    near_events = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        detections = []
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) != CAR_CLASS_ID:
                continue
            bb = box.cpu().numpy()
            cx, cy = bb_centroid(bb)
            detections.append({
                "bbox": bb,
                "cx": cx,
                "cy": cy
            })

        detections = tracker.update(detections, frame_idx)

        # SPEED CALCULATION
        for tr in tracker.tracks.values():
            if H is None or len(tr["hist"]) < 2:
                continue
            f0,x0,y0 = tr["hist"][-2]
            f1,x1,y1 = tr["hist"][-1]
            dt = (f1-f0)/fps
            if dt > 0:
                dist = np.hypot(x1-x0,y1-y0)
                tr["mph"] = (dist/dt)*2.23694

        # NEAR MISS DETECTION
        track_list = list(tracker.tracks.values())
        for i in range(len(track_list)):
            for j in range(i+1,len(track_list)):
                t1 = track_list[i]
                t2 = track_list[j]
                if len(t1["hist"])<2 or len(t2["hist"])<2:
                    continue

                _,x1,y1 = t1["hist"][-1]
                _,x2,y2 = t2["hist"][-1]
                dx,dy = x2-x1,y2-y1
                dist = np.hypot(dx,dy)

                if dist>near_dist_thresh:
                    continue

                v1 = np.array(t1["hist"][-1][1:]) - np.array(t1["hist"][-2][1:])
                v2 = np.array(t2["hist"][-1][1:]) - np.array(t2["hist"][-2][1:])
                rel_v = (v2-v1)*fps
                closing_speed = np.dot(rel_v, np.array([dx,dy])/(dist+1e-6))

                if closing_speed<0:
                    ttc = dist/abs(closing_speed)
                    if ttc<ttc_thresh:
                        near_miss_count+=1
                        near_events.append({
                            "frame":frame_idx,
                            "car1":t1["id"],
                            "car2":t2["id"],
                            "distance_m":round(dist,2),
                            "ttc_s":round(ttc,2)
                        })

                        cv2.line(frame,(int(t1["cx"]),int(t1["cy"])),
                                       (int(t2["cx"]),int(t2["cy"])),RED,3)

                        cv2.putText(frame,f"NEAR MISS TTC={ttc:.2f}s",
                                    (50,50),cv2.FONT_HERSHEY_SIMPLEX,
                                    1,RED,3)

        # DRAW VEHICLES
        for tr in tracker.tracks.values():
            x1,y1,x2,y2 = map(int,tr["bbox"])
            color = GREEN if tr["crossed"] else YELLOW
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            mph_text = f"{tr['mph']:.1f} mph" if tr["mph"] else "--"
            cv2.putText(frame,f"CAR {tr['id']} {mph_text}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        writer.write(frame)
        frame_idx+=1

    cap.release()
    writer.release()

    pd.DataFrame(near_events).to_csv("near_misses.csv",index=False)

    return output_path, near_miss_count
