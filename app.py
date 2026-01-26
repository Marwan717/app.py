# app.py — Alternate Dashboard Template (Cars only + forward line-cross + green after crossing)
# Cloud-safe: YOLO detect (.predict) + lightweight tracker (no lap/bytetrack).
#
# requirements.txt:
# streamlit
# ultralytics
# opencv-python-headless
# numpy
# pandas

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

def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def bb_centroid(bb):
    x1,y1,x2,y2 = bb
    return (x1+x2)/2.0, (y1+y2)/2.0

def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    area_a = max(0.0,ax2-ax1)*max(0.0,ay2-ay1)
    area_b = max(0.0,bx2-bx1)*max(0.0,by2-by1)
    denom = area_a + area_b - inter
    return float(inter/denom) if denom>0 else 0.0

def side_of_line(px, py, x1, y1, x2, y2):
    return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)

class CarTracker:
    def __init__(self, max_age=20, iou_thr=0.25, max_dist_px=80.0):
        self.max_age = max_age
        self.iou_thr = iou_thr
        self.max_dist = max_dist_px
        self.next_id = 1
        self.tracks = {}  # tid -> dict

    def _new_track(self, det):
        tid = self.next_id; self.next_id += 1
        self.tracks[tid] = {
            "id": tid,
            "bbox": det["bbox"],
            "cx": det["cx"], "cy": det["cy"],
            "prev_cx": det["cx"], "prev_cy": det["cy"],
            "prev_side": None,
            "last_frame": det["frame"],
            "crossed": False
        }
        return tid

    def update(self, dets, frame_idx):
        dead = [tid for tid,tr in self.tracks.items() if frame_idx - tr["last_frame"] > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        if not dets:
            return []

        track_ids = list(self.tracks.keys())
        unmatched = set(range(len(dets)))
        used = set()
        matches = []

        cand = []
        for di,d in enumerate(dets):
            for tid in track_ids:
                if tid in used:
                   
