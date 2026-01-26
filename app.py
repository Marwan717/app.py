# app.py — Streamlit Traffic Video Analytics (Cloud-safe) + mph on boxes
# - Drag & drop video
# - Yellow boxes + class + ID + SPEED (mph)
# - Lightweight tracker (no lap / no bytetrack)
# - Throttled live preview for smoother UI
# - Saves annotated mp4 (video-only) + optional audio remux (best-effort)
#
# requirements.txt:
# streamlit
# ultralytics
# opencv-python-headless
# numpy
# pandas

import os
import json
import time
import shutil
import tempfile
import subprocess
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# COCO ids
KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
YELLOW = (0, 255, 255)  # BGR


def make_writer(path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


def bb_centroid(bb):
    x1, y1, x2, y2 = bb
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


class SimpleTracker:
    """
    Cloud-safe tracker:
    - Greedy association using IoU then centroid distance
    - Maintains IDs, last bbox, last centroid, speed (EMA)
    """
    def __init__(self, fps: float, meters_per_pixel: float | None,
                 max_age_frames: int = 20,
                 iou_match_thresh: float = 0.25,
                 max_center_dist_px: float = 80.0,
                 speed_ema_alpha: float = 0.35):
        self.fps = fps
        self.mpp = meters_per_pixel
        self.max_age = max_age_frames
        self.iou_thr = iou_match_thresh
        self.max_dist = max_center_dist_px
        self.alpha = speed_ema_alpha

        self.next_id = 1
        self.tracks = {}  # id -> dict

    def _new_track(self, det):
        tid = self.next_id
        self.next_id += 1
        cx, cy = det["cx"], det["cy"]
        self.tracks[tid] = {
            "id": tid,
            "class": det["class"],
            "bbox": det["bbox"],
            "cx": cx,
            "cy": cy,
            "last_frame": det["frame"],
            "speed_mph": None,
            "speed_hist": deque(maxlen=10),
        }
        return tid

    def update(self, dets, frame_idx: int):
        """
        dets: list of dicts: {bbox, cx, cy, class, conf, frame}
        returns: list of dicts with assigned track_id and speed_mph
        """
        # Age-out old tracks
        to_del = []
        for tid, tr in self.tracks.items():
            if frame_idx - tr["last_frame"] > self.max_age:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        if not dets:
            return []

        track_ids = list(self.tracks.keys())

        # Greedy matching:
        # 1) Prefer same class matches
        # 2) Highest IoU
        # 3) Then centroid distance threshold
        unmatched_dets = set(range(len(dets)))
        used_tracks = set()

        matches = []

        # Build candidate pairs
        candidates = []
        for di, d in enumerate(dets):
            for tid in track_ids:
                tr = self.tracks[tid]
                if tr["class"] != d["class"]:
                    continue
                iou = iou_xyxy(tr["bbox"], d["bbox"])
                if iou < self.iou_thr:
                    continue
                # higher IoU = better
                candidates.append((iou, tid, di))

        # Sort by best IoU first
        candidates.sort(reverse=True, key=lambda x: x[0])

        for iou, tid, di in candidates:
            if di not in unmatched_dets or tid in used_tracks:
                continue
            # Accept match
            used_tracks.add(tid)
            unmatched_dets.remove(di)
            matches.append((tid, di))

        # For remaining dets, try centroid distance match (same class)
        for di in list(unmatched_dets):
            d = dets[di]
            best = None
            for tid in track_ids:
                if tid in used_tracks:
                    continue
                tr = self.tracks[tid]
                if tr["class"] != d["class"]:
                    continue
                dist = float(np.hypot(d["cx"] - tr["cx"], d["cy"] - tr["cy"]))
                if dist <= self.max_dist:
                    if best is None or dist < best[0]:
                        best = (dist, tid)
            if best is not None:
                _, tid = best
                used_tracks.add(tid)
                unmatched_dets.remove(di)
                matches.append((tid, di))

        # Update matched tracks
        for tid, di in matches:
            d = dets[di]
            tr = self.tracks[tid]

            # speed from centroid delta (if last seen previous frame-ish)
            dt_frames = max(1, d["frame"] - tr["last_frame"])
            dx = d["cx"] - tr["cx"]
            dy = d["cy"] - tr["cy"]
            dist_px = float(np.hypot(dx, dy))
            px_per_s = dist_px * (self.fps / dt_frames)

            mph = None
            if self.mpp and self.mpp > 0:
                mps = px_per_s * self.mpp
                mph = mps * 2.2369362921

                # EMA smoothing
                if tr["speed_mph"] is None:
                    tr["speed_mph"] = mph
                else:
                    tr["speed_mph"] = self.alpha * mph + (1 - self.alpha) * tr["speed_mph"]
                tr["speed_hist"].append(tr["speed_mph"])

            # Update track state
            tr["bbox"] = d["bbox"]
            tr["cx"] = d["cx"]
            tr["cy"] = d["cy"]
            tr["last_frame"] = d["frame"]

            d["track_id"] = tid
            d["speed_mph"] = tr["speed_mph"]

        # Create new tracks for unmatched detections
        for di in list(unmatched_dets):
            tid = self._new_track(dets[di])
            dets[di]["track_id"] = tid
            dets[di]["speed_mph"] = None

        # Return dets with IDs
        return dets


def try_remux_audio(original_video: str, annotated_video: str, out_with_audio: str) -> bool:
    """
    Best-effort: mux audio from original + video from annotated using ffmpeg if present.
    On many Streamlit Cloud images ffmpeg may not exist; then we just skip.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    cmd = [
        ffmpeg, "-y",
        "-i", annotated_video,
        "-i", original_video,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        out_with_audio
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return p.returncode == 0 and os.path.exists(out_with_audio) and os.path.getsize(out_with_audio) > 0
    except Exception:
        return False


def analyze_video(
    video_path: str,
    outdir: str,
    model_name: str,
    conf: float,
    iou: float,
    imgsz: int,
    max_frames: int,
    meters_per_pixel: float | None,
    downscale: float,
    preview_every_n: int,
    live_preview_slot,
    live_stats_slot,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    model = YOLO(model_name)

    # Optional downscale for speed
    w = int(w0 * downscale)
    h = int(h0 * downscale)
    if w <= 0 or h <= 0:
        w, h = w0, h0
        downscale = 1.0

    annotated_mp4 = os.path.join(outdir, "annotated_output.mp4")
    writer = make_writer(annotated_mp4, fps, w, h)

    # Tracker (mph needs mpp)
    tracker = SimpleTracker(
        fps=fps,
        meters_per_pixel=meters_per_pixel,
        max_age_frames=20,
        iou_match_thresh=0.25,
        max_center_dist_px=80.0 * downscale,
        speed_ema_alpha=0.35
    )

    # Logging
    rows = []
    unique_ids_by_class = defaultdict(set)
    totals_unique = defaultdict(int)  # derived later

    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if downscale != 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        # Detect (no lap dependency)
        res = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

        dets = []
        frame_counts = defaultdict(int)

        if res.boxes is not None and res.boxes.xyxy is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            for bb, c, cf in zip(xyxy, cls, confs):
                if c not in KEEP:
                    continue
                cname = KEEP[c]
                frame_counts[cname] += 1
                cx, cy = bb_centroid(bb)
                dets.append({
                    "frame": frame_idx,
                    "class": cname,
                    "conf": float(cf),
                    "bbox": bb.astype(float),
                    "cx": float(cx),
                    "cy": float(cy),
                })

        dets = tracker.update(dets, frame_idx)

        # Draw + record
        for d in dets:
            cname = d["class"]
            tid = d.get("track_id")
            if tid is not None:
                unique_ids_by_class[cname].add(int(tid))

            x1, y1, x2, y2 = map(int, d["bbox"].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 3)

            mph = d.get("speed_mph")
            if mph is not None:
                label = f"{cname} ID:{tid}  {mph:.1f} mph"
            else:
                label = f"{cname} ID:{tid}"

            cv2.putText(frame, label, (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

            rows.append({
                "frame": frame_idx,
                "track_id": tid,
                "class": cname,
                "conf": d["conf"],
                "x1": float(d["bbox"][0]),
                "y1": float(d["bbox"][1]),
                "x2": float(d["bbox"][2]),
                "y2": float(d["bbox"][3]),
                "cx": d["cx"],
                "cy": d["cy"],
                "speed_mph": float(mph) if mph is not None else None,
            })

        # HUD
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames if total_frames else '?'}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 2)

        # Write annotated frame
        writer.write(frame)

        # Update UI less often for smoothness
        if preview_every_n <= 1 or (frame_idx % preview_every_n == 0):
            live_preview_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    caption="Live preview (throttled)", use_container_width=True)

            live_stats_slot.markdown("### Live stats")
            live_stats_slot.write("**This frame**")
            live_stats_slot.json(dict(frame_counts))

            live_stats_slot.write("**Unique objects so far (IDs)**")
            live_stats_slot.json({k: len(v) for k, v in unique_ids_by_class.items()})

        frame_idx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t0

    # Save outputs
    df = pd.DataFrame(rows)
    detections_csv = os.path.join(outdir, "tracks_with_speed.csv")
    df.to_csv(detections_csv, index=False)

    counts = {k: len(v) for k, v in unique_ids_by_class.items()}
    counts["TOTAL_UNIQUE"] = int(sum(counts.values()))
    counts["processed_frames"] = frame_idx
    counts["elapsed_s"] = float(elapsed)
    counts["meters_per_pixel"] = meters_per_pixel
    counts["downscale"] = downscale
    counts["preview_every_n"] = preview_every_n

    counts_json = os.path.join(outdir, "counts.json")
    with open(counts_json, "w") as f:
        json.dump(counts, f, indent=2)

    # Best-effort audio remux
    annotated_with_audio = os.path.join(outdir, "annotated_with_audio.mp4")
    has_audio = try_remux_audio(video_path, annotated_mp4, annotated_with_audio)

    return {
        "annotated_video": annotated_mp4,
        "annotated_with_audio": annotated_with_audio if has_audio else None,
        "csv": detections_csv,
        "counts_json": counts_json,
        "df": df,
        "counts": counts,
        "has_audio_remux": has_audio,
    }


# --------------------------- STREAMLIT UI ---------------------------

st.set_page_config(page_title="Traffic Video Analytics (mph)", layout="wide")
st.title("Traffic Video Analytics — Yellow Boxes + Speed (mph)")
st.caption("Cloud-safe: no ByteTrack/lap. Uses a lightweight tracker to estimate mph (needs meters-per-pixel).")

# Top: Upload + settings
top_left, top_right = st.columns([2, 1])

with top_left:
    st.subheader("1) Upload")
    uploaded = st.file_uploader("Drag & drop a video", type=["mp4", "mov", "avi", "mkv"])

with top_right:
    st.subheader("2) Settings")
    model_name = st.selectbox("Model (faster = smoother)", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    conf = st.slider("Confidence", 0.05, 0.80, 0.20, 0.05)
    iou = st.slider("IoU", 0.10, 0.90, 0.50, 0.05)
    imgsz = st.select_slider("Image size", [640, 768, 896, 960, 1024], value=768)

    st.markdown("---")
    st.subheader("Speed calibration")
    st.caption("Enter meters-per-pixel (m/px) to show mph on boxes. If 0, mph will be blank.")
    mpp = st.number_input("Meters per pixel (m/px)", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
    meters_per_pixel = mpp if mpp > 0 else None

    st.markdown("---")
    st.subheader("Performance")
    downscale = st.selectbox("Downscale frames", [1.0, 0.75, 0.5], index=1)  # 0.75 default helps smoothness
    preview_every_n = st.selectbox("Update preview every N frames", [1, 2, 3, 5, 10], index=2)  # 3 default
    max_frames = st.number_input("Max frames (0=all)", min_value=0, value=300, step=100)

run = st.button("Run analysis", type="primary", disabled=(uploaded is None))

st.divider()

# Middle: Live preview + stats
mid_left, mid_right = st.columns([2, 1])
live_preview_slot = mid_left.empty()
live_stats_slot = mid_right.empty()

st.divider()
final_slot = st.container()

if run and uploaded is not None:
    workdir = tempfile.mkdtemp(prefix="traffic_mph_")
    in_path = os.path.join(workdir, uploaded.name)
    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success("Processing… (preview is throttled for smoother UI).")

    try:
        outputs = analyze_video(
            video_path=in_path,
            outdir=workdir,
            model_name=model_name,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_frames=int(max_frames),
            meters_per_pixel=meters_per_pixel,
            downscale=float(downscale),
            preview_every_n=int(preview_every_n),
            live_preview_slot=live_preview_slot,
            live_stats_slot=live_stats_slot,
        )

        c = outputs["counts"]
        with final_slot:
            st.success("Done ✅")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Frames", c.get("processed_frames", 0))
            k2.metric("Elapsed (s)", f"{c.get('elapsed_s', 0):.1f}")
            k3.metric("Unique objects (approx.)", c.get("TOTAL_UNIQUE", 0))
            k4.metric("mph enabled", "Yes" if meters_per_pixel else "No (set m/px)")

            st.subheader("Annotated video")
            st.video(outputs["annotated_video"])
            st.caption("OpenCV output is video-only. If audio remux succeeds, you’ll also get a version with audio below.")

            if outputs["annotated_with_audio"]:
                st.subheader("Annotated video (with original audio)")
                st.video(outputs["annotated_with_audio"])
            else:
                st.info("Audio remux not available in this environment. Download the video and your original audio will remain only in the original file.")

            st.subheader("Downloads")
            d1, d2, d3 = st.columns(3)
            with d1:
                with open(outputs["annotated_video"], "rb") as f:
                    st.download_button("Download annotated_output.mp4", f, file_name="annotated_output.mp4")
            with d2:
                with open(outputs["csv"], "rb") as f:
                    st.download_button("Download tracks_with_speed.csv", f, file_name="tracks_with_speed.csv")
            with d3:
                with open(outputs["counts_json"], "rb") as f:
                    st.download_button("Download counts.json", f, file_name="counts.json")

            if outputs["annotated_with_audio"]:
                with open(outputs["annotated_with_audio"], "rb") as f:
                    st.download_button("Download annotated_with_audio.mp4", f, file_name="annotated_with_audio.mp4")

            st.subheader("Counts (unique IDs)")
            st.json({k: v for k, v in c.items() if k not in ["processed_frames", "elapsed_s", "meters_per_pixel", "downscale", "preview_every_n"]})

            st.subheader("Sample tracks (first 200 rows)")
            st.dataframe(outputs["df"].head(200), use_container_width=True)

            st.info(
                "Smoothness tips: use yolov8n.pt, downscale=0.5 or 0.75, preview every 3–5 frames, and cap max_frames while testing."
            )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a video and click Run. For smoother performance: use yolov8n.pt + downscale 0.75 + preview every 3 frames.")
