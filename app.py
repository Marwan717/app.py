# app.py — Streamlit Stable Traffic Analytics
# Calibrated MPH + Line Count + Near Miss + TTC

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

YELLOW = (0,255,255)
GREEN  = (0,255,0)
RED    = (0,0,255)
CYAN   = (255,255,0)
WHITE  = (255,255,255)

# ---------------------------
# Utility
# ---------------------------

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

# ---------------------------
# Simple Tracker
# ---------------------------

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
                if np.hypot(d["cx"]-tr["cx"], d["cy"]-tr["cy"])<50:
                    tr["prev"]=tr["curr"]
                    tr["curr"]=(d["cx"],d["cy"])
                    tr["bbox"]=d["bbox"]
                    tr["last"]=frame_idx

                    if self.H:
                        gx,gy=project_point(self.H,d["cx"],d["cy"])
                        tr["hist"].append((frame_idx,gx,gy))

                    d["id"]=tid
                    matched=True
                    break

            if not matched:
                tid=self.next_id
                self.next_id+=1
                hist=deque(maxlen=30)
                if self.H:
                    gx,gy=project_point(self.H,d["cx"],d["cy"])
                    hist.append((frame_idx,gx,gy))

                self.tracks[tid]={
                    "bbox":d["bbox"],
                    "curr":(d["cx"],d["cy"]),
                    "prev":(d["cx"],d["cy"]),
                    "hist":hist,
                    "mph":None,
                    "crossed":False,
                    "last":frame_idx
                }
                d["id"]=tid
        return dets

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(layout="wide")
st.title("Traffic Analytics")

uploaded = st.file_uploader("Upload video", type=["mp4","mov","avi","mkv"])

with st.sidebar:
    st.header("Settings")

    conf = st.slider("Confidence",0.1,0.8,0.3,0.05)
    imgsz = st.selectbox("Image size",[640,768],index=0)
    downscale = st.selectbox("Downscale",[1.0,0.75,0.5],index=2)
    frame_skip = st.slider("Frame skip",1,5,2)

    st.subheader("Line Count")
    line_pos = st.slider("Line height (%)",0.1,0.9,0.6)

    st.subheader("Calibration")
    enable_calib = st.checkbox("Enable speed calibration",True)
    rect_w = st.number_input("Road width (m)",value=3.7)
    rect_l = st.number_input("Road length (m)",value=10.0)

    p1x=st.number_input("P1 x",value=100)
    p1y=st.number_input("P1 y",value=100)
    p2x=st.number_input("P2 x",value=300)
    p2y=st.number_input("P2 y",value=100)
    p3x=st.number_input("P3 x",value=300)
    p3y=st.number_input("P3 y",value=300)
    p4x=st.number_input("P4 x",value=100)
    p4y=st.number_input("P4 y",value=300)

    st.subheader("Near Miss")
    enable_nm = st.checkbox("Enable near miss detection",True)
    dist_thresh = st.slider("Conflict distance (m)",1.0,10.0,4.0)
    ttc_thresh = st.slider("TTC threshold (s)",0.5,5.0,2.0)

run = st.button("Run analysis", disabled=(uploaded is None))

preview = st.empty()

if run and uploaded is not None:

    workdir=tempfile.mkdtemp()
    path=os.path.join(workdir,uploaded.name)
    with open(path,"wb") as f:
        f.write(uploaded.getbuffer())

    cap=cv2.VideoCapture(path)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    w=int(w*downscale)
    h=int(h*downscale)

    model=YOLO("yolov8n.pt")

    H=None
    if enable_calib:
        px=[(p1x*downscale,p1y*downscale),
            (p2x*downscale,p2y*downscale),
            (p3x*downscale,p3y*downscale),
            (p4x*downscale,p4y*downscale)]
        H=build_homography(px,rect_w,rect_l)

    tracker=Tracker(fps,H)

    line_y=int(line_pos*h)

    near_events=[]
    forward_count=0
    frame_idx=0

    while True:
        ret,frame=cap.read()
        if not ret: break

        if frame_idx % frame_skip !=0:
            frame_idx+=1
            continue

        frame=cv2.resize(frame,(w,h))

        results=model.predict(frame,conf=conf,imgsz=imgsz,verbose=False)[0]

        dets=[]
        for box,cls in zip(results.boxes.xyxy,results.boxes.cls):
            if int(cls)!=CAR_CLASS_ID: continue
            bb=box.cpu().numpy()
            cx,cy=bb_centroid(bb)
            dets.append({"bbox":bb,"cx":cx,"cy":cy})

        dets=tracker.update(dets,frame_idx)

        cv2.line(frame,(0,line_y),(w,line_y),CYAN,2)

        # Speed
        for tid,tr in tracker.tracks.items():
            if H and len(tr["hist"])>=2:
                f0,x0,y0=tr["hist"][-2]
                f1,x1,y1=tr["hist"][-1]
                dt=(f1-f0)/fps
                if dt>0:
                    dist=np.hypot(x1-x0,y1-y0)
                    tr["mph"]=(dist/dt)*2.23694

        # Line count
        for tid,tr in tracker.tracks.items():
            if not tr["crossed"]:
                if tr["prev"][1] < line_y and tr["curr"][1]>=line_y:
                    tr["crossed"]=True
                    forward_count+=1

        # Near Miss
        conflict_ids=set()
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
                            conflict_ids.add(id1)
                            conflict_ids.add(id2)
                            near_events.append({
                                "time_s":round(frame_idx/fps,2),
                                "car1":id1,
                                "car2":id2,
                                "distance_m":round(dist,2),
                                "ttc_s":round(ttc,2)
                            })

        # Draw
        for tid,tr in tracker.tracks.items():
            x1,y1,x2,y2=map(int,tr["bbox"])
            color=RED if tid in conflict_ids else (GREEN if tr["crossed"] else YELLOW)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            speed_txt=f"{tr['mph']:.1f} mph" if tr["mph"] else ""
            cv2.putText(frame,f"ID {tid} {speed_txt}",(x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        cv2.putText(frame,f"Count {forward_count}",(20,h-20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,WHITE,2)

        preview.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                      use_container_width=True)

        frame_idx+=1

    cap.release()

    st.success("Done")
    st.metric("Forward count",forward_count)
    st.write("Near Miss Events")
    st.dataframe(pd.DataFrame(near_events))
