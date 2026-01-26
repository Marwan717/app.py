# app.py — Cooler UI + Better Speed (mph) via optional 4-point calibration
# Cloud-safe (NO lap / NO bytetrack). Uses YOLO detect + simple ID tracking.
#
# requirements.txt:
# streamlit
# ultralytics
# opencv-python-headless
# numpy
# pandas

import os, json, time, tempfile, shutil, subprocess
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# COCO ids
KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
YELLOW = (0, 255, 255)  # BGR


def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


def bb_centroid(bb):
    x1, y1, x2, y2 = bb
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def iou_xyxy(a, b):
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


def try_remux_audio(original_video, annotated_video, out_with_audio):
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


class SpeedProjector:
    """
    Converts pixel centroids -> (x_m, y_m) on ground plane.
    Two modes:
      - quick: meters_per_pixel (approx)
      - calibrated: 4-point homography to a rectangle of known width/length (recommended)
    """
    def __init__(self, meters_per_pixel=None, H=None):
        self.mpp = meters_per_pixel
        self.H = H  # 3x3 homography from image(px) -> ground(m)

    def has_calibration(self):
        return self.H is not None or (self.mpp is not None and self.mpp > 0)

    def project(self, cx, cy):
        # returns (gx, gy) in meters (or None if uncalibrated)
        if self.H is not None:
            pt = np.array([[[cx, cy]]], dtype=np.float32)  # shape (1,1,2)
            gp = cv2.perspectiveTransform(pt, self.H)[0][0]
            return float(gp[0]), float(gp[1])
        if self.mpp is not None and self.mpp > 0:
            return float(cx * self.mpp), float(cy * self.mpp)
        return None


class SimpleTracker:
    """
    Lightweight Cloud-safe tracker (no lap):
    - Greedy association using IoU then centroid distance.
    - Per-track history of ground-projected positions to compute stable speed.
    """
    def __init__(self, fps, projector: SpeedProjector,
                 max_age_frames=20, iou_thr=0.25, max_dist_px=80.0,
                 speed_window_frames=15, speed_clip_mph=120.0):
        self.fps = fps
        self.projector = projector
        self.max_age = max_age_frames
        self.iou_thr = iou_thr
        self.max_dist = max_dist_px
        self.speed_window = max(3, int(speed_window_frames))
        self.speed_clip = float(speed_clip_mph)

        self.next_id = 1
        self.tracks = {}  # id -> dict

    def _new_track(self, det):
        tid = self.next_id
        self.next_id += 1
        gxgy = self.projector.project(det["cx"], det["cy"])
        self.tracks[tid] = {
            "id": tid,
            "class": det["class"],
            "bbox": det["bbox"],
            "cx": det["cx"],
            "cy": det["cy"],
            "last_frame": det["frame"],
            "history": deque(maxlen=120),  # store (frame, gx, gy) if calibrated
            "speed_mph": None
        }
        if gxgy is not None:
            self.tracks[tid]["history"].append((det["frame"], gxgy[0], gxgy[1]))
        return tid

    def _compute_speed_mph(self, tr):
        # Uses a window over history in meters to reduce jitter.
        if len(tr["history"]) < 2:
            return None
        # take last N points
        hist = list(tr["history"])[-self.speed_window:]
        if len(hist) < 2:
            return None
        f0, x0, y0 = hist[0]
        f1, x1, y1 = hist[-1]
        dt = (f1 - f0) / self.fps
        if dt <= 1e-6:
            return None
        dist_m = float(np.hypot(x1 - x0, y1 - y0))
        mps = dist_m / dt
        mph = mps * 2.2369362921
        # clamp unrealistic spikes
        if mph < 0 or mph > self.speed_clip:
            return None
        return mph

    def update(self, dets, frame_idx):
        # age-out
        to_del = [tid for tid, tr in self.tracks.items() if frame_idx - tr["last_frame"] > self.max_age]
        for tid in to_del:
            del self.tracks[tid]

        if not dets:
            return []

        track_ids = list(self.tracks.keys())
        unmatched = set(range(len(dets)))
        used_tracks = set()
        matches = []

        # 1) IoU match same-class
        candidates = []
        for di, d in enumerate(dets):
            for tid in track_ids:
                tr = self.tracks[tid]
                if tr["class"] != d["class"]:
                    continue
                iou = iou_xyxy(tr["bbox"], d["bbox"])
                if iou >= self.iou_thr:
                    candidates.append((iou, tid, di))
        candidates.sort(reverse=True, key=lambda x: x[0])

        for _, tid, di in candidates:
            if di not in unmatched or tid in used_tracks:
                continue
            used_tracks.add(tid)
            unmatched.remove(di)
            matches.append((tid, di))

        # 2) centroid distance match same-class
        for di in list(unmatched):
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
                unmatched.remove(di)
                matches.append((tid, di))

        # update matched
        for tid, di in matches:
            d = dets[di]
            tr = self.tracks[tid]

            tr["bbox"] = d["bbox"]
            tr["cx"] = d["cx"]
            tr["cy"] = d["cy"]
            tr["last_frame"] = d["frame"]

            gxgy = self.projector.project(d["cx"], d["cy"])
            if gxgy is not None:
                tr["history"].append((d["frame"], gxgy[0], gxgy[1]))
                mph = self._compute_speed_mph(tr)
                tr["speed_mph"] = mph
            else:
                tr["speed_mph"] = None

            d["track_id"] = tid
            d["speed_mph"] = tr["speed_mph"]

        # new tracks for unmatched
        for di in list(unmatched):
            tid = self._new_track(dets[di])
            dets[di]["track_id"] = tid
            dets[di]["speed_mph"] = self.tracks[tid]["speed_mph"]

        return dets


def build_homography(px_points, rect_width_m, rect_length_m):
    """
    Map 4 pixel points (clockwise) to a ground-plane rectangle:
      (0,0), (W,0), (W,L), (0,L)
    """
    src = np.array(px_points, dtype=np.float32)
    dst = np.array([[0, 0], [rect_width_m, 0], [rect_width_m, rect_length_m], [0, rect_length_m]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)  # image->ground
    return H


def analyze_video(
    video_path, outdir, model_name, conf, iou, imgsz,
    max_frames, downscale, preview_every_n,
    tracker_cfg, projector: SpeedProjector,
    live_preview_slot, live_stats_slot
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    model = YOLO(model_name)

    w = int(w0 * downscale)
    h = int(h0 * downscale)
    if w <= 0 or h <= 0:
        w, h = w0, h0
        downscale = 1.0

    annotated_mp4 = os.path.join(outdir, "annotated_output.mp4")
    writer = make_writer(annotated_mp4, fps, w, h)

    tracker = SimpleTracker(
        fps=fps,
        projector=projector,
        max_age_frames=tracker_cfg["max_age"],
        iou_thr=tracker_cfg["iou_thr"],
        max_dist_px=tracker_cfg["max_dist_px"] * downscale,
        speed_window_frames=tracker_cfg["speed_window"],
        speed_clip_mph=tracker_cfg["speed_clip_mph"],
    )

    rows = []
    unique_ids_by_class = defaultdict(set)

    t0 = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if downscale != 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

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

        for d in dets:
            cname = d["class"]
            tid = d.get("track_id")
            mph = d.get("speed_mph")

            if tid is not None:
                unique_ids_by_class[cname].add(int(tid))

            x1, y1, x2, y2 = map(int, d["bbox"].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 3)

            # Show mph only when calibrated
            if mph is not None:
                label = f"{cname}  #{tid}  {mph:.1f} mph"
            else:
                label = f"{cname}  #{tid}"

            cv2.putText(frame, label, (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, YELLOW, 2)

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

        cv2.putText(frame, f"Frame {frame_idx}/{total_frames if total_frames else '?'}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 2)

        writer.write(frame)

        if preview_every_n <= 1 or (frame_idx % preview_every_n == 0):
            live_preview_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    use_container_width=True)

            live_stats_slot.markdown("### Live dashboard")
            live_stats_slot.metric("Frame", frame_idx)
            live_stats_slot.write("**This frame counts**")
            live_stats_slot.json(dict(frame_counts))
            live_stats_slot.write("**Unique IDs so far**")
            live_stats_slot.json({k: len(v) for k, v in unique_ids_by_class.items()})

        frame_idx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "tracks_with_speed.csv")
    df.to_csv(csv_path, index=False)

    counts = {k: len(v) for k, v in unique_ids_by_class.items()}
    counts["TOTAL_UNIQUE"] = int(sum(counts.values()))
    counts["processed_frames"] = frame_idx
    counts["elapsed_s"] = float(elapsed)

    meta = {
        "calibrated": projector.H is not None,
        "meters_per_pixel": projector.mpp,
    }

    counts_json = os.path.join(outdir, "counts.json")
    with open(counts_json, "w") as f:
        json.dump({"counts": counts, "meta": meta}, f, indent=2)

    # best-effort audio remux
    annotated_with_audio = os.path.join(outdir, "annotated_with_audio.mp4")
    has_audio = try_remux_audio(video_path, annotated_mp4, annotated_with_audio)

    return {
        "annotated_video": annotated_mp4,
        "annotated_with_audio": annotated_with_audio if has_audio else None,
        "csv": csv_path,
        "counts_json": counts_json,
        "df": df,
        "counts": counts,
        "meta": meta,
        "has_audio_remux": has_audio,
    }


# ---------------------------- STREAMLIT UI ----------------------------

st.set_page_config(page_title="Traffic Analytics (Cool UI + mph)", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem;}
    .small-note {opacity: 0.8; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚦 Proactive Traffic Video Analytics")
st.markdown('<div class="small-note">Yellow boxes + IDs + smoother mph estimates (best with 4-point calibration).</div>',
            unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    conf = st.slider("Confidence", 0.05, 0.80, 0.20, 0.05)
    iou = st.slider("IoU", 0.10, 0.90, 0.50, 0.05)
    imgsz = st.select_slider("Image size", [640, 768, 896, 960, 1024], value=768)

    st.divider()
    st.subheader("Speed accuracy mode")

    mode = st.radio("Calibration", ["Quick (approx)", "Calibrated (recommended)"], index=1)

    meters_per_pixel = None
    H = None

    if mode == "Quick (approx)":
        st.caption("Approximate: one scale for whole image. Often inaccurate due to perspective.")
        mpp = st.number_input("Meters per pixel (m/px)", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
        meters_per_pixel = mpp if mpp > 0 else None

    else:
        st.caption("Better: map 4 road points to a real rectangle (width x length in meters).")
        rect_w = st.number_input("Rectangle width (m)", min_value=0.1, value=3.7, step=0.1)
        rect_l = st.number_input("Rectangle length (m)", min_value=0.1, value=10.0, step=0.5)

        st.write("Enter 4 pixel points (clockwise):")
        c1, c2 = st.columns(2)
        with c1:
            x1 = st.number_input("P1 x", min_value=0, value=100)
            y1 = st.number_input("P1 y", min_value=0, value=100)
            x2 = st.number_input("P2 x", min_value=0, value=300)
            y2 = st.number_input("P2 y", min_value=0, value=100)
        with c2:
            x3 = st.number_input("P3 x", min_value=0, value=300)
            y3 = st.number_input("P3 y", min_value=0, value=300)
            x4 = st.number_input("P4 x", min_value=0, value=100)
            y4 = st.number_input("P4 y", min_value=0, value=300)

        # Homography will be built after we know downscale (we apply it later if needed)
        px_points_user = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    st.divider()
    st.subheader("Performance")
    downscale = st.selectbox("Downscale", [1.0, 0.75, 0.5], index=1)
    preview_every_n = st.selectbox("Preview update every N frames", [1, 2, 3, 5, 10], index=2)
    max_frames = st.number_input("Max frames (0=all)", min_value=0, value=300, step=100)

    st.divider()
    st.subheader("Tracker tuning (simple)")
    speed_window = st.slider("Speed smoothing window (frames)", 5, 45, 15, 5)
    speed_clip_mph = st.slider("Max mph clamp", 30, 150, 120, 10)
    max_age = st.slider("Track timeout (frames)", 5, 60, 20, 5)

# Main layout
tab_run, tab_help = st.tabs(["▶ Run", "🛠 Fix speed accuracy"])

with tab_run:
    top = st.container()
    with top:
        colA, colB = st.columns([2, 1])
        with colA:
            uploaded = st.file_uploader("Drop a video file", type=["mp4", "mov", "avi", "mkv"])
        with colB:
            st.markdown("#### Output")
            st.write("• Annotated video")
            st.write("• CSV tracks")
            st.write("• Counts JSON")

    run = st.button("Run analysis", type="primary", disabled=(uploaded is None))

    st.divider()

    live_left, live_right = st.columns([2, 1])
    live_preview_slot = live_left.empty()
    live_stats_slot = live_right.empty()

    st.divider()
    final_slot = st.container()

    if run and uploaded is not None:
        workdir = tempfile.mkdtemp(prefix="traffic_cool_")
        in_path = os.path.join(workdir, uploaded.name)
        with open(in_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Build projector
        if mode == "Calibrated (recommended)":
            # Adjust points for downscale
            px_points = [(int(px * downscale), int(py * downscale)) for (px, py) in px_points_user]
            H = build_homography(px_points, float(rect_w), float(rect_l))
            projector = SpeedProjector(H=H)
        else:
            projector = SpeedProjector(meters_per_pixel=meters_per_pixel)

        tracker_cfg = {
            "speed_window": int(speed_window),
            "speed_clip_mph": float(speed_clip_mph),
            "max_age": int(max_age),
            "iou_thr": 0.25,
            "max_dist_px": 80.0,
        }

        with st.spinner("Processing… (preview updates are throttled for smoothness)"):
            outputs = analyze_video(
                video_path=in_path,
                outdir=workdir,
                model_name=model_name,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_frames=int(max_frames),
                downscale=float(downscale),
                preview_every_n=int(preview_every_n),
                tracker_cfg=tracker_cfg,
                projector=projector,
                live_preview_slot=live_preview_slot,
                live_stats_slot=live_stats_slot,
            )

        c = outputs["counts"]
        meta = outputs["meta"]

        with final_slot:
            st.success("Done ✅")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Frames", c.get("processed_frames", 0))
            k2.metric("Elapsed (s)", f"{c.get('elapsed_s', 0):.1f}")
            k3.metric("Unique objects", c.get("TOTAL_UNIQUE", 0))
            k4.metric("Speed mode", "Calibrated" if meta["calibrated"] else ("Quick" if meta["meters_per_pixel"] else "Off"))

            st.subheader("Annotated video")
            st.video(outputs["annotated_video"])

            if outputs["annotated_with_audio"]:
                st.caption("Audio preserved (ffmpeg remux succeeded).")
                st.video(outputs["annotated_with_audio"])
            else:
                st.caption("Audio may not be preserved in this environment (OpenCV writes video-only).")

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

            st.subheader("Counts (unique IDs)")
            st.json({k: v for k, v in c.items() if k not in ["processed_frames", "elapsed_s"]})

            st.subheader("Speed sanity check (distribution)")
            df = outputs["df"]
            if "speed_mph" in df.columns and df["speed_mph"].notna().any():
                s = df["speed_mph"].dropna()
                st.write(f"Median mph: **{s.median():.1f}**, 90th percentile: **{s.quantile(0.9):.1f}**")
                hist = np.histogram(s.values, bins=20)
                st.bar_chart(pd.DataFrame({"count": hist[0]}, index=[f"{hist[1][i]:.0f}-{hist[1][i+1]:.0f}" for i in range(len(hist[0]))]))
            else:
                st.warning("No mph computed. Enable calibration (quick m/px or calibrated 4-point).")

with tab_help:
    st.markdown(
        """
        ## Why the mph looked wrong
        Most traffic videos have **perspective**. One “meters-per-pixel” number makes cars far away appear slower/faster incorrectly.

        ## Best way to improve mph fast
        Use **Calibrated (recommended)**:
        1) Pick a rectangle on the road you can measure (or estimate well)  
           - width: lane width (~3.7 m)  
           - length: a marked distance (e.g., 10 m between lane markers/crosswalk stripes)
        2) Enter the 4 pixel points clockwise around that rectangle.
        3) Run again.

        You’ll instantly get more believable mph because the math happens on the **ground plane**.

        ## Quick checks
        - If cars show 80–120 mph in a city intersection: calibration is wrong.
        - If speeds are too “spiky”: increase **Speed smoothing window**.
        - If IDs jump: increase **Track timeout** slightly.
        """
    )
