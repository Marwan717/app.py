# app.py
# Streamlit Traffic Video Analytics (Cloud/GitHub-safe)
# - Drag & drop video
# - Yellow detection boxes + labels
# - Simple, always-visible stats
# - Live preview while processing
# - Saves annotated video + detections CSV + counts JSON
#
# IMPORTANT:
# - Uses YOLOv8 .predict() (NOT .track()) to avoid "No module named 'lap'" on Streamlit Cloud.
# - This version is detection-only (no persistent IDs, no tracking-based speed).
#
# Install:
#   pip install streamlit ultralytics opencv-python-headless numpy pandas
# Run:
#   streamlit run app.py

import os
import json
import time
import tempfile
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# COCO class IDs (YOLOv8 default COCO)
KEEP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Yellow in OpenCV uses BGR
YELLOW = (0, 255, 255)


def make_writer(path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


def analyze_video_detection_only(
    video_path: str,
    outdir: str,
    model_name: str,
    conf: float,
    iou: float,
    imgsz: int,
    max_frames: int,
    live_preview_slot,
    live_stats_slot,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    model = YOLO(model_name)

    annotated_mp4 = os.path.join(outdir, "annotated_output.mp4")
    writer = make_writer(annotated_mp4, fps, w, h)

    # Outputs
    rows = []
    total_counts = defaultdict(int)

    t0 = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        frame_counts = defaultdict(int)

        # DETECTION ONLY: no ByteTrack, no lap dependency
        res = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

        if res.boxes is not None and res.boxes.xyxy is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            for bb, c, cf in zip(xyxy, cls, confs):
                if c not in KEEP:
                    continue

                cname = KEEP[c]
                frame_counts[cname] += 1
                total_counts[cname] += 1

                x1, y1, x2, y2 = map(int, bb.tolist())

                # Yellow rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 3)

                # Label (simple)
                label = f"{cname}  conf:{cf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    YELLOW,
                    2,
                )

                rows.append(
                    {
                        "frame": frame_idx,
                        "class": cname,
                        "conf": float(cf),
                        "x1": float(bb[0]),
                        "y1": float(bb[1]),
                        "x2": float(bb[2]),
                        "y2": float(bb[3]),
                    }
                )

        # HUD
        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total_frames if total_frames else '?'}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            YELLOW,
            2,
        )

        # Save frame
        writer.write(frame)

        # LIVE PREVIEW in Streamlit
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        live_preview_slot.image(rgb, use_container_width=True)

        # LIVE STATS in Streamlit (simple + obvious)
        live_stats_slot.markdown("### Live stats")
        live_stats_slot.write("**This frame**")
        live_stats_slot.json(dict(frame_counts))
        live_stats_slot.write("**Totals (detections, not unique objects)**")
        live_stats_slot.json(dict(total_counts))

        frame_idx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    detections_csv = os.path.join(outdir, "detections.csv")
    df.to_csv(detections_csv, index=False)

    counts = dict(total_counts)
    counts["processed_frames"] = frame_idx
    counts["elapsed_s"] = float(elapsed)

    counts_json = os.path.join(outdir, "counts.json")
    with open(counts_json, "w") as f:
        json.dump(counts, f, indent=2)

    return {
        "annotated_video": annotated_mp4,
        "detections_csv": detections_csv,
        "counts_json": counts_json,
        "df": df,
        "counts": counts,
    }


# --------------------------- STREAMLIT UI (NEW LAYOUT) ---------------------------

st.set_page_config(page_title="Traffic Video Analytics (Simple)", layout="wide")

# Top bar
st.title("Traffic Video Analytics — Simple (Streamlit Cloud Safe)")
st.caption("Drop a video → see yellow detections live → download annotated output + CSV.")

# Layout: 3 zones
# 1) Upload + controls (top)
# 2) Live view + stats (middle, side-by-side)
# 3) Final results + downloads (bottom)

with st.container():
    up_col, settings_col = st.columns([2, 1])

    with up_col:
        st.subheader("1) Upload")
        uploaded = st.file_uploader("Drag & drop a traffic video", type=["mp4", "mov", "avi", "mkv"])

    with settings_col:
        st.subheader("2) Settings")
        model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
        conf = st.slider("Confidence", 0.05, 0.80, 0.20, 0.05)
        iou = st.slider("IoU", 0.10, 0.90, 0.50, 0.05)
        imgsz = st.select_slider("Image size", [640, 768, 896, 960, 1024, 1280], value=960)
        max_frames = st.number_input("Max frames (0 = all)", min_value=0, value=300, step=100)

run = st.button("Run analysis", type="primary", disabled=(uploaded is None))

st.divider()

# Middle: Live preview + stats
live_left, live_right = st.columns([2, 1])
live_preview_slot = live_left.empty()
live_stats_slot = live_right.empty()

st.divider()

# Bottom: Final results
final_slot = st.container()

if run and uploaded is not None:
    workdir = tempfile.mkdtemp(prefix="traffic_simple_")
    in_path = os.path.join(workdir, uploaded.name)

    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success("Processing started. You should see yellow boxes in the live preview above.")

    try:
        outputs = analyze_video_detection_only(
            video_path=in_path,
            outdir=workdir,
            model_name=model_name,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_frames=int(max_frames),
            live_preview_slot=live_preview_slot,
            live_stats_slot=live_stats_slot,
        )

        c = outputs["counts"]

        with final_slot:
            st.success("Done ✅")

            k1, k2, k3 = st.columns(3)
            k1.metric("Frames processed", c.get("processed_frames", 0))
            k2.metric("Elapsed (s)", f"{c.get('elapsed_s', 0):.1f}")
            k3.metric("Total detections", sum(v for k, v in c.items() if k not in ["processed_frames", "elapsed_s"]))

            st.subheader("Final totals (detections, not unique objects)")
            st.json({k: v for k, v in c.items() if k not in ["processed_frames", "elapsed_s"]})

            st.subheader("Annotated video")
            st.video(outputs["annotated_video"])
            st.caption("If preview is blank on Streamlit Cloud, use the download button below.")

            d1, d2, d3 = st.columns(3)
            with d1:
                with open(outputs["annotated_video"], "rb") as f:
                    st.download_button("Download annotated_output.mp4", f, file_name="annotated_output.mp4")
            with d2:
                with open(outputs["detections_csv"], "rb") as f:
                    st.download_button("Download detections.csv", f, file_name="detections.csv")
            with d3:
                with open(outputs["counts_json"], "rb") as f:
                    st.download_button("Download counts.json", f, file_name="counts.json")

            st.subheader("Detections table (sample)")
            st.dataframe(outputs["df"].head(300), use_container_width=True)

            st.info(
                "This version is detection-only (no tracking IDs) so totals are *detections*, "
                "not unique vehicle counts. It’s the simplest Cloud-safe prototype."
            )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a video, then click **Run analysis**. Live preview + stats will appear above.")
