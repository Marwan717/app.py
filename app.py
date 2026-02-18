# app.py — Traffic Analytics + Calibrated MPH + Near Miss Detection

import os, json, time, tempfile
from collections import deque
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

CAR_CLASS_ID = 2

YELLOW = (0,255,255)
GREEN  = (0,255,0)
CYAN   = (255,255,0)
WHITE  = (255,255,255)
RED    = (0,0,255)

# -------------------------------------------------------
# Utility
# -------------------------------------------------------

def bb_centroid(bb):
    x1,y1,x2,y2 = bb
    return (x1+x2)/2,(y1+y2)/2

def build_homography(px_points, rect_w, rect_l):
    src = np.array(px_points, dtype=np.float32)
    dst = np.array([[0,0],[rect_w,0],[rect_w,rect_l],[0,rect_l]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src,dst)

def project_point(H,cx,cy):
    pt = np.array([[[cx,cy]]],dtype=np.float32)
    gp = cv2.perspectiveTransform(pt,H)[0][0]
    return float(gp[0]),float(gp[1])

# -------------------------------------------------------
# Tracker
# -------------------------------------------------------

class Tracker:
    def __init__(self,fps,H=None):
        self.fps=fps
        self.H=H
        self.tracks={}
        self.next_id=1

    def update(self,dets,frame_idx):
        for d in dets:
            matched=False
            for tid,tr in self.tracks.items():
                if np.hypot(d["cx"]-tr["cx"], d["cy"]-tr["cy"])<60:
                    tr["bbox"]=d["bbox"]
                    tr["prev"]=tr["curr"]
                    tr["curr"]=(d["cx"],d["cy"])
                    tr["last"]=frame_idx
                    if self.H:
                        gx,gy=project_point(self.H,d["cx"],d["cy"])
                        tr["hist"].append((frame_idx,gx,gy))
                    d["id"]=tid
                    matched=True
                    break
            if not matched:
                tid=self.next_id; self.next_id+=1
                hist=deque(maxlen=60)
                if self.H:
                    gx,gy=project_point(self.H,d["cx"],d["cy"])
                    hist.append((frame_idx,gx,gy))
                self.tracks[tid]={
                    "bbox":d["bbox"],
                    "curr":(d["cx"],d["cy"]),
                    "prev":(d["cx"],d["cy"]),
                    "hist":hist,
                    "mph":None,
                    "last":frame_idx
                }
                d["id"]=tid
        return dets

# -------------------------------------------------------
# Analysis
# -------------------------------------------------------

def analyze(video_path,H,enable_nm,dist_thresh,ttc_thresh):

    cap=cv2.VideoCapture(video_path)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model=YOLO("yolov8n.pt")
    tracker=Tracker(fps,H)

    out="annotated.mp4"
    writer=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))

    near_events=[]
    frame_idx=0

    while True:
        ret,frame=cap.read()
        if not ret: break

        results=model(frame,verbose=False)[0]
        dets=[]

        for box,cls in zip(results.boxes.xyxy,results.boxes.cls):
            if int(cls)!=CAR_CLASS_ID: continue
            bb=box.cpu().numpy()
            cx,cy=bb_centroid(bb)
            dets.append({"bbox":bb,"cx":cx,"cy":cy})

        dets=tracker.update(dets,frame_idx)

        # speed
        for tr in tracker.tracks.values():
            if H and len(tr["hist"])>=2:
                f0,x0,y0=tr["hist"][-2]
                f1,x1,y1=tr["hist"][-1]
                dt=(f1-f0)/fps
                if dt>0:
                    dist=np.hypot(x1-x0,y1-y0)
                    tr["mph"]=(dist/dt)*2.23694

        # near miss
        if enable_nm and H:
            tracks=list(tracker.tracks.items())
            for i in range(len(tracks)):
                for j in range(i+1,len(tracks)):
                    id1,t1=tracks[i]
                    id2,t2=tracks[j]
                    if len(t1["hist"])<2 or len(t2["hist"])<2: continue

                    _,x1,y1=t1["hist"][-1]
                    _,x2,y2=t2["hist"][-1]
                    dx,dy=x2-x1,y2-y1
                    dist=np.hypot(dx,dy)
                    if dist>dist_thresh: continue

                    v1=np.array(t1["hist"][-1][1:])-np.array(t1["hist"][-2][1:])
                    v2=np.array(t2["hist"][-1][1:])-np.array(t2["hist"][-2][1:])
                    rel_v=(v2-v1)*fps
                    closing=np.dot(rel_v,np.array([dx,dy])/(dist+1e-6))

                    if closing<0:
                        ttc=dist/abs(closing)
                        if ttc<ttc_thresh:
                            near_events.append({
                                "frame":frame_idx,
                                "car1":id1,
                                "car2":id2,
                                "distance_m":round(dist,2),
                                "ttc_s":round(ttc,2)
                            })

                            cv2.line(frame,
                                     (int(t1["curr"][0]),int(t1["curr"][1])),
                                     (int(t2["curr"][0]),int(t2["curr"][1])),
                                     RED,3)

                            cv2.putText(frame,
                                        f"NEAR MISS TTC={ttc:.2f}s",
                                        (40,40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,RED,3)

        # draw cars
        for tid,tr in tracker.tracks.items():
            x1,y1,x2,y2=map(int,tr["bbox"])
            cv2.rectangle(frame,(x1,y1),(x2,y2),Y
