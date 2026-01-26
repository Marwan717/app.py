# app.py
# Drag-and-drop traffic video analytics prototype (single file)
#
# Features:
# - Drop box to upload video
# - Detect + track road users (vehicles + pedestrians + bikes) using YOLOv8 + ByteTrack
# - Classify and count unique objects by type
# - Estimate speed (m/s & mph) if you provide a meters-per-pixel scale; otherwise shows px/s
# - Outputs: annotated video + per-frame tracks CSV + per-object summary CSV + counts JSON
#
# Install:
#   pip install streamlit ultralytics opencv-python numpy pandas numpy
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


# COCO classes we care about
COCO_KEEP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

TYPE_GROUP = {
    "person": "pedestrian",
    "bicycle": "bike",
    "motorcycle": "motorcycle",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
}


def centroid_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def write_video_from_frames(out_path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))


def analyze_video(
    video_path: str,
    outdir: str,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 960,
    device=None,
    meters_per_pixel: float | None = None,
    save_trails: bool = True,
    trail_len: int = 60,
    max_frames: int = 0,
    progress_cb=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    model = YOLO(model_name)

    annotated_path = os.path.join(outdir, "annotated_output.mp4")
    writer = write_video_from_frames(annotated_path, fps, w, h)

    track_rows = []
    obj_class = {}
    obj_group = {}
    obj_first_frame = {}
    obj_last_frame = {}
    obj_points = defaultdict(list)
    obj_speeds = defaultdict(list)
    obj_dists = defaultdict(float)
    unique_by_class = defaultdict(set)
    trails = defaultdict(list)

    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if max_frames and frame_idx >= max_frames:
            break

        # YOLOv8 tracking (ByteTrack)
        results = model.track(
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes

        # overlay info
        frame_counts = defaultdict(int)

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            for tid, c, confv, bb in zip(ids, cls, confs, xyxy):
                if c not in COCO_KEEP:
                    continue

                cname = COCO_KEEP[c]
                gname = TYPE_GROUP.get(cname, cname)

                frame_counts[cname] += 1
                unique_by_class[cname].add(tid)

                obj_class.setdefault(tid, cname)
                obj_group.setdefault(tid, gname)
                obj_first_frame.setdefault(tid, frame_idx)
                obj_last_frame[tid] = frame_idx

                cx, cy = centroid_xyxy(bb)
                obj_points[tid].append((frame_idx, cx, cy))

                trails[tid].append((int(cx), int(cy)))
                if len(trails[tid]) > trail_len:
                    trails[tid] = trails[tid][-trail_len:]

                # speed from last two points
                speed_val = None
                if len(obj_points[tid]) >= 2:
                    _, x_prev, y_prev = obj_points[tid][-2]
                    dx = cx - x_prev
                    dy = cy - y_prev
                    dist_px = float(np.hypot(dx, dy))
                    px_per_sec = dist_px * fps

                    if meters_per_pixel is not None and meters_per_pixel > 0:
                        dist_m = dist_px * meters_per_pixel
                        obj_dists[tid] += dist_m
                        speed_mps = px_per_sec * meters_per_pixel
                        speed_val = speed_mps
                        obj_speeds[tid].append(speed_mps)
                    else:
                        obj_dists[tid] += dist_px
                        speed_val = px_per_sec
                        obj_speeds[tid].append(px_per_sec)

                # record per-frame
                track_rows.append(
                    {
                        "frame": frame_idx,
                        "track_id": tid,
                        "class": cname,
                        "group": gname,
                        "conf": float(confv),
                        "x1": float(bb[0]),
                        "y1": float(bb[1]),
                        "x2": float(bb[2]),
                        "y2": float(bb[3]),
                        "cx": float(cx),
                        "cy": float(cy),
                        "speed_mps": float(speed_val) if (speed_val is not None and meters_per_pixel) else None,
                        "speed_pxps": float(speed_val) if (speed_val is not None and not meters_per_pixel) else None,
                    }
                )

                # draw bbox + label
                x1, y1, x2, y2 = map(int, bb.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{cname} ID:{tid}"
                if speed_val is not None and meters_per_pixel:
                    mph = speed_val * 2.2369362921
                    label += f" {speed_val:.1f} m/s ({mph:.1f} mph)"
                elif speed_val is not None:
                    label += f" {speed_val:.1f} px/s"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # draw trails
        if save_trails:
            for tid, pts in trails.items():
                if len(pts) >= 2:
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i - 1], pts[i], (255, 255, 0), 2)

        # HUD
        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total_frames if total_frames else '?'}  FPS:{fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        y = 60
        for cname in ["person", "bicycle", "motorcycle", "car", "bus", "truck"]:
            if cname in frame_counts:
                cv2.putText(
                    frame,
                    f"{cname}: {frame_counts[cname]}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                y += 24

        writer.write(frame)

        frame_idx += 1
        if progress_cb:
            # keep it safe when total_frames is unknown
            denom = max(total_frames, frame_idx, 1)
            progress_cb(min(frame_idx / denom, 1.0), frame_idx)

    cap.release()
    writer.release()

    elapsed = time.time() - t0

    # Save data outputs
    tracks_csv = os.path.join(outdir, "tracks_per_frame.csv")
    df_tracks = pd.DataFrame(track_rows)
    df_tracks.to_csv(tracks_csv, index=False)

    summaries = []
    for tid, cname in obj_class.items():
        speeds = obj_speeds.get(tid, [])
        avg_speed = float(np.mean(speeds)) if speeds else None
        max_speed = float(np.max(speeds)) if speeds else None

        summaries.append(
            {
                "track_id": tid,
                "class": cname,
                "group": obj_group.get(tid, cname),
                "first_frame": obj_first_frame.get(tid),
                "last_frame": obj_last_frame.get(tid),
                "duration_s": ((obj_last_frame.get(tid, 0) - obj_first_frame.get(tid, 0)) / fps)
                if tid in obj_first_frame
                else None,
                "distance_m": obj_dists[tid] if (meters_per_pixel and meters_per_pixel > 0) else None,
                "distance_px": obj_dists[tid] if not (meters_per_pixel and meters_per_pixel > 0) else None,
                "avg_speed_mps": avg_speed if (meters_per_pixel and meters_per_pixel > 0) else None,
                "max_speed_mps": max_speed if (meters_per_pixel and meters_per_pixel > 0) else None,
                "avg_speed_pxps": avg_speed if not (meters_per_pixel and meters_per_pixel > 0) else None,
                "max_speed_pxps": max_speed if not (meters_per_pixel and meters_per_pixel > 0) else None,
            }
        )

    summary_csv = os.path.join(outdir, "object_summary.csv")
    df_sum = pd.DataFrame(summaries).sort_values(by=["class", "track_id"])
    df_sum.to_csv(summary_csv, index=False)

    counts = {c: len(ids) for c, ids in unique_by_class.items()}
    counts["TOTAL_UNIQUE"] = int(sum(counts.get(c, 0) for c in COCO_KEEP.values()))
    counts["note"] = "Counts are unique track IDs (approx. unique objects)."
    counts["meters_per_pixel"] = meters_per_pixel
    counts["processed_frames"] = frame_idx
    counts["elapsed_s"] = elapsed
    counts_path = os.path.join(outdir, "counts.json")
    with open(counts_path, "w") as f:
        json.dump(counts, f, indent=2)

    return {
        "annotated_video": annotated_path,
        "tracks_csv": tracks_csv,
        "summary_csv": summary_csv,
        "counts_json": counts_path,
        "df_tracks": df_tracks,
        "df_summary": df_sum,
        "counts": counts,
    }


# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="Traffic Video Analytics Prototype", layout="wide")

st.title("Proactive Traffic Safety — Video Analytics Prototype")
st.caption("Drop a video → detect & track road users → counts + speed + trajectories → outputs & tables.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    conf = st.slider("Confidence", 0.05, 0.80, 0.25, 0.05)
    iou = st.slider("IoU (NMS)", 0.10, 0.90, 0.50, 0.05)
    imgsz = st.select_slider("Image size", options=[640, 768, 896, 960, 1024, 1280], value=960)

    st.markdown("---")
    st.subheader("Speed scale (optional)")
    st.write("If you know meters-per-pixel, enter it. Otherwise speed will be px/s.")
    meters_per_pixel = st.number_input("Meters per pixel", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
    meters_per_pixel = meters_per_pixel if meters_per_pixel > 0 else None

    st.markdown("---")
    save_trails = st.checkbox("Draw trajectory trails", value=True)
    trail_len = st.slider("Trail length (points)", 10, 200, 60, 10)

    st.markdown("---")
    max_frames = st.number_input("Max frames (0 = all)", min_value=0, value=0, step=50)


st.header("1) Drop your video here")
uploaded = st.file_uploader("Drag & drop a video file (mp4/mov/avi)", type=["mp4", "mov", "avi", "mkv"])

run_btn = st.button("Run analysis", type="primary", disabled=(uploaded is None))

st.header("2) Results")

if run_btn and uploaded is not None:
    workdir = tempfile.mkdtemp(prefix="traffic_analytics_")
    in_path = os.path.join(workdir, uploaded.name)

    # Save uploaded file to disk
    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success(f"Uploaded: {uploaded.name}")
    st.video(in_path)

    prog = st.progress(0)
    status = st.empty()

    def progress_cb(p, frame_idx):
        prog.progress(int(p * 100))
        status.write(f"Processing… frame {frame_idx}")

    try:
        with st.spinner("Running detection + tracking + speed/trajectory estimation…"):
            outputs = analyze_video(
                video_path=in_path,
                outdir=workdir,
                model_name=model_name,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                meters_per_pixel=meters_per_pixel,
                save_trails=save_trails,
                trail_len=trail_len,
                max_frames=int(max_frames),
                progress_cb=progress_cb,
            )

        prog.progress(100)
        status.write("Done ✅")

        # Summary KPIs
        c = outputs["counts"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Unique objects (approx.)", c.get("TOTAL_UNIQUE", 0))
        col2.metric("Frames processed", c.get("processed_frames", 0))
        col3.metric("Elapsed (s)", f"{c.get('elapsed_s', 0):.1f}")
        col4.metric("Speed units", "m/s" if c.get("meters_per_pixel") else "px/s")

        # Tabs for "another section"
        tab1, tab2, tab3 = st.tabs(["Summary", "Tables", "Annotated video"])

        with tab1:
            st.subheader("Counts (unique track IDs)")
            counts_table = {k: v for k, v in c.items() if k in ["person", "bicycle", "motorcycle", "car", "bus", "truck", "TOTAL_UNIQUE"]}
            st.json(counts_table)

            st.subheader("Downloads")
            with open(outputs["counts_json"], "rb") as f:
                st.download_button("Download counts.json", f, file_name="counts.json")

            with open(outputs["tracks_csv"], "rb") as f:
                st.download_button("Download tracks_per_frame.csv", f, file_name="tracks_per_frame.csv")

            with open(outputs["summary_csv"], "rb") as f:
                st.download_button("Download object_summary.csv", f, file_name="object_summary.csv")

        with tab2:
            st.subheader("Object summary (unique IDs)")
            st.dataframe(outputs["df_summary"], use_container_width=True, height=350)

            st.subheader("Per-frame tracks (sample)")
            st.dataframe(outputs["df_tracks"].head(500), use_container_width=True, height=350)

        with tab3:
            st.subheader("Annotated output")
            st.video(outputs["annotated_video"])
            with open(outputs["annotated_video"], "rb") as f:
                st.download_button("Download annotated_output.mp4", f, file_name="annotated_output.mp4")

        st.info(
            "Note: Counts are based on unique tracking IDs (approx. unique objects). "
            "Speed is only meaningful if you provided a valid meters-per-pixel scale."
        )

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    st.write("Drop a video above, then click **Run analysis** to see results here.")
