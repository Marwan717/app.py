# app.py — Traffic Analytics + Calibrated MPH + Near Miss Detection

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


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def bb_centroid(bb):
    x1,y1,x2,y2 = bb
    return (x1+x2)/2,(y1+y2)/2


def build_homography(px_points, rect_w, rect_l):
    src = np.array(px_points, dtype=np.float32)
    dst = np.array([
        [0,0],
        [rect_w,0],
        [rect_w,rect_l],
        [0,rect_l]
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src,dst)


def project_point(H,cx,cy):
    pt = np.array([[[cx,cy]]],dtype=np.float32)
    gp = cv2.perspectiveTransform(pt,H)[0][0]
    return float(gp[0]),float(gp[1])


def iou(a,b):
    ax1,ay1,ax2,ay2=a
    bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1)
    ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw=max(0,ix2-ix1)
    ih=max(0,iy2-iy1)
    inter=iw*ih
    if inter<=0: return 0
    area_a=(ax2-ax1)*(ay2-ay1)
    area_b=(bx2-bx1)*(by2-by1)
    return inter/(area_a+area_b-inter+1e-6)


# -------------------------------------------------------
# Tracker
# -------------------------------------------------------

class Tracker:

    def __init__(self,fps,H=None):
        self.fps=float(fps)
        self.H=H
        self.tracks={}
        self.next_id=1


    def update(self,dets,frame_idx):

        # remove stale
        for tid in list(self.tracks.keys()):
            if frame_idx - self.tracks[tid]["last"] > 20:
                del self.tracks[tid]

        for d in dets:
            matched=False

            for tid,tr in self.tracks.items():

                dist=np.hypot(d["cx"]-tr["cx"], d["cy"]-tr["cy"])
                if dist<60:
                    tr["bbox"]=d["bbox"]
                    tr["prev"]=tr["curr"]
                    tr["curr"]=(d["cx"],d["cy"])
                    tr["cx"]=d["cx"]
                    tr["cy"]=d["cy"]
                    tr["last"]=frame_idx

                    if self.H is not None:
                        gx,gy=project_point(self.H,d["cx"],d["cy"])
                        tr["hist"].append((frame_idx,gx,gy))

                    d["id"]=tid
                    matched=True
                    break

            if not matched:
                tid=self.next_id
                self.next_id+=1

                hist=deque(maxlen=30)

                if self.H is not None:
                    gx,gy=project_point(self.H,d["cx"],d["cy"])
                    hist.append((frame_idx,gx,gy))

                self.tracks[tid]={
                    "bbox":d["bbox"],
                    "curr":(d["cx"],d["cy"]),
                    "prev":(d["cx"],d["cy"]),
                    "cx":d["cx"],
                    "cy":d["cy"],
                    "hist":hist,
                    "mph":None,
                    "crossed":False,
                    "last":frame_idx
                }

                d["id"]=tid

        return dets


# -------------------------------------------------------
# Analysis
# -------------------------------------------------------

def analyze(video_path,
            H,
            enable_nm,
            dist_thresh,
            ttc_thresh,
            line_y_frac):

    cap=cv2.VideoCapture(video_path)

    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    line_y=int(line_y_frac*h)

    model=YOLO("yolov8n.pt")
    tracker=Tracker(fps,H)

    out_path="annotated.mp4"
    writer=cv2.VideoWriter(out_path,
                           cv2.VideoWriter_fourcc(*"mp4v"),
                           fps,(w,h))

    near_events=[]
    frame_idx=0
    forward_count=0

    while True:

        ret,frame=cap.read()
        if not ret:
            break

        results=model(frame,verbose=False)[0]

        dets=[]
        if results.boxes is not None:
            for box,cls in zip(results.boxes.xyxy,
                               results.boxes.cls):

                if int(cls)!=CAR_CLASS_ID:
                    continue

                bb=box.cpu().numpy()
                cx,cy=bb_centroid(bb)

                dets.append({
                    "bbox":bb,
                    "cx":cx,
                    "cy":cy
                })

        dets=tracker.update(dets,frame_idx)

        # compute speed
        for tid,tr in tracker.tracks.items():
            if H is not None and len(tr["hist"])>=2:
                f0,x0,y0=tr["hist"][-2]
                f1,x1,y1=tr["hist"][-1]
                dt=(f1-f0)/fps
                if dt>0:
                    dist=np.hypot(x1-x0,y1-y0)
                    tr["mph"]=(dist/dt)*2.23694


        conflict_ids=set()

        # Near miss
        if enable_nm and H is not None:

            track_list=list(tracker.tracks.items())

            for i in range(len(track_list)):
                for j in range(i+1,len(track_list)):

                    id1,t1=track_list[i]
                    id2,t2=track_list[j]

                    if len(t1["hist"])<2 or len(t2["hist"])<2:
                        continue

                    _,x1,y1=t1["hist"][-1]
                    _,x2,y2=t2["hist"][-1]

                    dx,dy=x2-x1,y2-y1
                    dist=np.hypot(dx,dy)

                    if dist>dist_thresh:
                        continue

                    v1=np.array(t1["hist"][-1][1:])-np.array(t1["hist"][-2][1:])
                    v2=np.array(t2["hist"][-1][1:])-np.array(t2["hist"][-2][1:])
                    rel_v=(v2-v1)*fps

                    closing=np.dot(rel_v,
                                   np.array([dx,dy])/(dist+1e-6))

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
                                "ttc_s":round(ttc,2),
                                "speed1_mph":round(t1["mph"],1) if t1["mph"] else None,
                                "speed2_mph":round(t2["mph"],1) if t2["mph"] else None
                            })


        # Draw count line
        cv2.line(frame,(0,line_y),(w,line_y),CYAN,2)

        # Draw cars
        for tid,tr in tracker.tracks.items():

            x1,y1,x2,y2=map(int,tr["bbox"])
            color=RED if tid in conflict_ids else YELLOW

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

            mph_txt=f"{tr['mph']:.1f} mph" if tr["mph"] else "--"
            cv2.putText(frame,
                        f"ID {tid} {mph_txt}",
                        (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,color,2)

        writer.write(frame)
        frame_idx+=1

    cap.release()
    writer.release()

    near_df=pd.DataFrame(near_events)

    return out_path,near_df,forward_count


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

st.title("Traffic Analytics")

uploaded=st.file_uploader("Upload video",
                          type=["mp4","mov","avi","mkv"])

enable_calib=st.checkbox("Enable Speed Calibration")
enable_nm=st.checkbox("Enable Near Miss Detection")

line_y_frac=st.slider("Count line position",
                      0.1,0.9,0.5,0.01)

dist_thresh=st.slider("Near Miss Distance (m)",
                      1.0,10.0,5.0,0.5)

ttc_thresh=st.slider("TTC Threshold (s)",
                     0.5,5.0,2.0,0.5)

run=st.button("Run analysis")


if run and uploaded:

    temp_dir=tempfile.mkdtemp()
    video_path=os.path.join(temp_dir,uploaded.name)

    with open(video_path,"wb") as f:
        f.write(uploaded.read())

    H=None

    if enable_calib:
        # simple fixed rectangle example
        px_points=[(100,100),(300,100),(300,300),(100,300)]
        H=build_homography(px_points,3.7,10.0)

    with st.spinner("Processing..."):
        out_path,near_df,count=analyze(
            video_path,
            H,
            enable_nm,
            dist_thresh,
            ttc_thresh,
            line_y_frac
        )

    st.success("Done")

    st.video(out_path)

    st.subheader("Near Miss Events")
    st.dataframe(near_df)
