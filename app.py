# app.py — Alternate Template + Calibrated MPH + Concise Outputs
# Cars only • forward line crossing • box turns GREEN after crossing
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

CAR_CLASS_ID = 2  # COCO car

# OpenCV BGR colors
YELLOW = (0, 255, 255)  # before counted
GREEN  = (0, 255, 0)    # counted
CYAN   = (255, 255, 0)  # line
WHITE  = (255, 255, 255)

# ------------------------- Utils -------------------------

def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def bb_centroid(bb):
    x1, y1, x2, y2 = bb
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

def side_of_line(px, py, x1, y1, x2, y2):
    # sign of cross product
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

def build_homography(px_points, rect_width_m, rect_length_m):
    """
    px_points: 4 pixel points clockwise around a rectangle on the road.
    Maps image -> ground plane meters rectangle:
      (0,0), (W,0), (W,L), (0,L)
    """
    src = np.array(px_points, dtype=np.float32)
    dst = np.array([[0, 0], [rect_width_m, 0], [rect_width_m, rect_length_m], [0, rect_length_m]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H

def project_point(H, cx, cy):
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    gp = cv2.perspectiveTransform(pt, H)[0][0]
    return float(gp[0]), float(gp[1])

# ------------------------- Tracker (with calibrated speed) -------------------------

class CarTracker:
    """
    Lightweight tracker:
      - Greedy IoU then centroid distance
      - Keeps 'crossed' state so boxes stay green after counting
      - Stores projected ground points to compute smoothed mph
    """
    def __init__(self, fps, H=None, speed_window=12, max_age=20, iou_thr=0.25, max_dist_px=80.0, mph_clip=120.0):
        self.fps = float(fps)
        self.H = H  # homography image->ground meters (or None)
        self.speed_window = int(max(3, speed_window))
        self.max_age = int(max_age)
        self.iou_thr = float(iou_thr)
        self.max_dist = float(max_dist_px)
        self.mph_clip = float(mph_clip)

        self.next_id = 1
        self.tracks = {}  # tid -> dict

    def _new_track(self, det):
        tid = self.next_id
        self.next_id += 1

        tr = {
            "id": tid,
            "bbox": det["bbox"],
            "cx": det["cx"], "cy": det["cy"],
            "prev_cx": det["cx"], "prev_cy": det["cy"],
            "prev_side": None,
            "last_frame": det["frame"],
            "crossed": False,

            # speed
            "hist_m": deque(maxlen=120),  # (frame_idx, gx, gy)
            "mph": None
        }

        if self.H is not None:
            gx, gy = project_point(self.H, det["cx"], det["cy"])
            tr["hist_m"].append((det["frame"], gx, gy))

        self.tracks[tid] = tr
        return tid

    def _compute_mph(self, tr):
        if self.H is None:
            return None
        if len(tr["hist_m"]) < 2:
            return None
        hist = list(tr["hist_m"])[-self.speed_window:]
        if len(hist) < 2:
            return None

        f0, x0, y0 = hist[0]
        f1, x1, y1 = hist[-1]
        dt = (f1 - f0) / self.fps
        if dt <= 1e-6:
            return None
        dist_m = float(np.hypot(x1 - x0, y1 - y0))
        mph = (dist_m / dt) * 2.2369362921
        if mph < 0 or mph > self.mph_clip:
            return None
        return mph

    def update(self, dets, frame_idx):
        # remove stale tracks
        dead = [tid for tid, tr in self.tracks.items() if frame_idx - tr["last_frame"] > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        if not dets:
            return []

        track_ids = list(self.tracks.keys())
        unmatched = set(range(len(dets)))
        used = set()
        matches = []

        # IoU candidates
        cand = []
        for di, d in enumerate(dets):
            for tid in track_ids:
                if tid in used:
                    continue
                tr = self.tracks[tid]
                iou = iou_xyxy(tr["bbox"], d["bbox"])
                if iou >= self.iou_thr:
                    cand.append((iou, tid, di))
        cand.sort(reverse=True, key=lambda x: x[0])

        for _, tid, di in cand:
            if di not in unmatched or tid in used:
                continue
            used.add(tid)
            unmatched.remove(di)
            matches.append((tid, di))

        # centroid fallback
        for di in list(unmatched):
            d = dets[di]
            best = None
            for tid in track_ids:
                if tid in used:
                    continue
                tr = self.tracks[tid]
                dist = float(np.hypot(d["cx"] - tr["cx"], d["cy"] - tr["cy"]))
                if dist <= self.max_dist and (best is None or dist < best[0]):
                    best = (dist, tid)
            if best is not None:
                _, tid = best
                used.add(tid)
                unmatched.remove(di)
                matches.append((tid, di))

        # update matched
        for tid, di in matches:
            d = dets[di]
            tr = self.tracks[tid]

            tr["bbox"] = d["bbox"]
            tr["prev_cx"], tr["prev_cy"] = tr["cx"], tr["cy"]
            tr["cx"], tr["cy"] = d["cx"], d["cy"]
            tr["last_frame"] = d["frame"]

            if self.H is not None:
                gx, gy = project_point(self.H, tr["cx"], tr["cy"])
                tr["hist_m"].append((frame_idx, gx, gy))
                tr["mph"] = self._compute_mph(tr)
            else:
                tr["mph"] = None

            d["track_id"] = tid
            d["crossed"] = tr["crossed"]
            d["mph"] = tr["mph"]

        # new tracks
        for di in list(unmatched):
            tid = self._new_track(dets[di])
            dets[di]["track_id"] = tid
            dets[di]["crossed"] = self.tracks[tid]["crossed"]
            dets[di]["mph"] = self.tracks[tid]["mph"]

        return dets

# ------------------------- Core Analysis -------------------------

def analyze(
    video_path, outdir,
    model_name, conf, iou, imgsz,
    max_frames, downscale, preview_every_n,
    line_orientation, line_pos_frac, forward_direction,
    H, speed_window,
    tracker_cfg,
    live_img_slot, live_metrics_slot, progress_bar
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    w = int(w0 * downscale)
    h = int(h0 * downscale)
    if w <= 0 or h <= 0:
        w, h = w0, h0
        downscale = 1.0

    model = YOLO(model_name)

    annotated = os.path.join(outdir, "annotated_output.mp4")
    writer = make_writer(annotated, fps, w, h)

    tracker = CarTracker(
        fps=fps,
        H=H,
        speed_window=speed_window,
        max_age=tracker_cfg["max_age"],
        iou_thr=tracker_cfg["iou_thr"],
        max_dist_px=tracker_cfg["max_dist_px"] * downscale,
        mph_clip=tracker_cfg["mph_clip"],
    )

    # define count line
    if line_orientation == "Horizontal":
        y_line = int(line_pos_frac * h)
        x1, y1, x2, y2 = 0, y_line, w - 1, y_line
        x_line = None
    else:
        x_line = int(line_pos_frac * w)
        x1, y1, x2, y2 = x_line, 0, x_line, h - 1
        y_line = None

    forward_crossings = 0
    crossing_events = []  # concise event log rows

    # for speed stats (concise + presentable)
    mph_recent = deque(maxlen=400)

    t0 = time.time()
    fidx = 0
    last_ui_time = time.time()
    last_ui_frame = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and fidx >= max_frames:
            break

        if downscale != 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        res = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

        dets = []
        if res.boxes is not None and res.boxes.xyxy is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            for bb, c, cf in zip(xyxy, cls, confs):
                if c != CAR_CLASS_ID:
                    continue
                cx, cy = bb_centroid(bb)
                dets.append({
                    "frame": fidx,
                    "conf": float(cf),
                    "bbox": bb.astype(float),
                    "cx": float(cx),
                    "cy": float(cy),
                })

        dets = tracker.update(dets, fidx)

        # draw count line
        cv2.line(frame, (x1, y1), (x2, y2), CYAN, 3)
        cv2.putText(frame, f"COUNT LINE • Forward: {forward_direction}", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2)

        cars_in_frame = 0
        counted_visible = 0

        for d in dets:
            cars_in_frame += 1
            tid = int(d["track_id"])
            tr = tracker.tracks.get(tid)
            if tr is None:
                continue

            # update side sign
            side_now = side_of_line(tr["cx"], tr["cy"], x1, y1, x2, y2)
            side_prev = tr["prev_side"]
            tr["prev_side"] = side_now

            crossed_this_frame = False

            # detect sign flip (crossing)
            if side_prev is not None and (side_prev == 0 or side_now == 0 or (side_prev > 0) != (side_now > 0)):
                if line_orientation == "Horizontal":
                    if forward_direction == "Down":
                        moved_forward = (tr["prev_cy"] < y_line and tr["cy"] >= y_line)
                    else:
                        moved_forward = (tr["prev_cy"] > y_line and tr["cy"] <= y_line)
                else:
                    if forward_direction == "Right":
                        moved_forward = (tr["prev_cx"] < x_line and tr["cx"] >= x_line)
                    else:
                        moved_forward = (tr["prev_cx"] > x_line and tr["cx"] <= x_line)

                if moved_forward and not tr["crossed"]:
                    tr["crossed"] = True
                    forward_crossings += 1
                    crossed_this_frame = True

                    crossing_events.append({
                        "frame": fidx,
                        "time_s": round(fidx / float(fps), 3),
                        "car_id": tid,
                        "mph_at_cross": (round(tr["mph"], 1) if tr["mph"] is not None else None),
                        "conf": round(float(d["conf"]), 3),
                    })

            # speed stats collection
            if tr["mph"] is not None:
                mph_recent.append(float(tr["mph"]))

            color = GREEN if tr["crossed"] else YELLOW
            if tr["crossed"]:
                counted_visible += 1

            x1b, y1b, x2b, y2b = map(int, tr["bbox"].tolist())
            cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), color, 3)

            mph_txt = f"{tr['mph']:.1f} mph" if tr["mph"] is not None else ("mph: —" if H is not None else "mph: (calibrate)")
            label = f"CAR #{tid} • {mph_txt}" + (" • COUNTED" if tr["crossed"] else "")
            cv2.putText(frame, label, (x1b, max(24, y1b - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if crossed_this_frame:
                cv2.putText(frame, "+1", (x2b + 6, y1b + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 3)

        # bottom HUD
        cv2.putText(frame, f"FORWARD CAR COUNT: {forward_crossings}", (12, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, WHITE, 2)

        writer.write(frame)

        # UI updates
        if preview_every_n <= 1 or (fidx % preview_every_n == 0):
            live_img_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            now = time.time()
            dt = max(1e-6, now - last_ui_time)
            fps_ui = (fidx - last_ui_frame) / dt
            last_ui_time = now
            last_ui_frame = fidx

            # speed summary for status
            if len(mph_recent) >= 5:
                s = np.array(mph_recent, dtype=float)
                mph_med = float(np.median(s))
                mph_p90 = float(np.quantile(s, 0.90))
            else:
                mph_med = None
                mph_p90 = None

            live_metrics_slot.metric("Forward crossings", forward_crossings)
            live_metrics_slot.caption("Cars only • Green = counted")

            c1, c2 = live_metrics_slot.columns(2)
            c1.metric("Cars in frame", cars_in_frame)
            c2.metric("Counted visible", counted_visible)

            c3, c4 = live_metrics_slot.columns(2)
            c3.metric("Median mph", f"{mph_med:.1f}" if mph_med is not None else "—")
            c4.metric("p90 mph", f"{mph_p90:.1f}" if mph_p90 is not None else "—")

            live_metrics_slot.metric("Processing FPS (approx)", f"{fps_ui:.1f}")

            if total_frames > 0:
                progress_bar.progress(min(1.0, fidx / total_frames))

        fidx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t0

    # concise outputs
    crossings_df = pd.DataFrame(crossing_events)
    crossings_csv = os.path.join(outdir, "crossings.csv")
    crossings_df.to_csv(crossings_csv, index=False)

    # summary
    mph_vals = [x for x in mph_recent if x is not None]
    if len(mph_vals) >= 5:
        s = np.array(mph_vals, dtype=float)
        mph_summary = {
            "median_mph": float(np.median(s)),
            "p90_mph": float(np.quantile(s, 0.90)),
        }
    else:
        mph_summary = {"median_mph": None, "p90_mph": None}

    summary = {
        "forward_crossings": int(forward_crossings),
        "processed_frames": int(fidx),
        "elapsed_s": float(elapsed),
        "line_orientation": line_orientation,
        "line_pos_frac": float(line_pos_frac),
        "forward_direction": forward_direction,
        "speed_calibrated": bool(H is not None),
        "speed_window_frames": int(speed_window),
        "speed_summary": mph_summary,
        "events_rows": int(len(crossing_events)),
    }

    summary_json = os.path.join(outdir, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    return annotated, crossings_csv, summary_json, crossings_df, summary

# ------------------------- UI (Alternate Dashboard Template) -------------------------

st.set_page_config(page_title="Car Counter + Calibrated MPH", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.1rem;}
.hero {
  padding: 16px 18px;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(13,20,40,0.92), rgba(10,14,22,0.92));
  border: 1px solid rgba(255,255,255,0.08);
}
.hero-title {font-size: 26px; font-weight: 800; margin: 0;}
.hero-sub {opacity: 0.82; margin-top: 6px; font-size: 0.95rem;}
.card {
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(15,23,42,0.60);
  border: 1px solid rgba(255,255,255,0.08);
}
.small {opacity: 0.78; font-size: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-title">🚗 Car Counter • Forward Line • Calibrated MPH</div>
  <div class="hero-sub">Cars only. Cross forward → counted (box turns green). Speed uses road-plane calibration (4 points).</div>
</div>
""", unsafe_allow_html=True)

st.write("")

controls, live, metrics = st.columns([1.10, 2.10, 1.05])

with controls:
    st.markdown('<div class="card"><b>1) Upload</b></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")

    st.write("")
    st.markdown('<div class="card"><b>2) Detection</b></div>', unsafe_allow_html=True)
    model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"], index=0)
    conf = st.slider("Confidence", 0.05, 0.80, 0.20, 0.05)
    iou = st.slider("IoU", 0.10, 0.90, 0.50, 0.05)
    imgsz = st.select_slider("Image size", [640, 768, 896, 960], value=768)

    st.write("")
    st.markdown('<div class="card"><b>3) Count line</b><div class="small">Place where cars pass cleanly.</div></div>', unsafe_allow_html=True)
    line_orientation = st.selectbox("Orientation", ["Horizontal", "Vertical"], index=0)
    line_pos_frac = st.slider("Position", 0.10, 0.90, 0.55, 0.01)
    if line_orientation == "Horizontal":
        forward_direction = st.selectbox("Forward direction", ["Down", "Up"], index=0)
    else:
        forward_direction = st.selectbox("Forward direction", ["Right", "Left"], index=0)

    st.write("")
    st.markdown('<div class="card"><b>4) Speed calibration (recommended)</b><div class="small">4 points clockwise around a road rectangle.</div></div>', unsafe_allow_html=True)

    use_calib = st.checkbox("Enable calibrated mph", value=True)

    rect_w = st.number_input("Rectangle width (m)", min_value=0.1, value=3.7, step=0.1, disabled=not use_calib)
    rect_l = st.number_input("Rectangle length (m)", min_value=0.1, value=10.0, step=0.5, disabled=not use_calib)

    c1, c2 = st.columns(2)
    with c1:
        p1x = st.number_input("P1 x", min_value=0, value=100, disabled=not use_calib)
        p1y = st.number_input("P1 y", min_value=0, value=100, disabled=not use_calib)
        p2x = st.number_input("P2 x", min_value=0, value=300, disabled=not use_calib)
        p2y = st.number_input("P2 y", min_value=0, value=100, disabled=not use_calib)
    with c2:
        p3x = st.number_input("P3 x", min_value=0, value=300, disabled=not use_calib)
        p3y = st.number_input("P3 y", min_value=0, value=300, disabled=not use_calib)
        p4x = st.number_input("P4 x", min_value=0, value=100, disabled=not use_calib)
        p4y = st.number_input("P4 y", min_value=0, value=300, disabled=not use_calib)

    speed_window = st.slider("Speed smoothing (frames)", 5, 45, 12, 5, disabled=not use_calib)

    st.write("")
    st.markdown('<div class="card"><b>5) Performance + Tracker</b></div>', unsafe_allow_html=True)
    downscale = st.selectbox("Downscale", [1.0, 0.75, 0.5], index=1)
    preview_every_n = st.selectbox("Preview every N frames", [1, 2, 3, 5, 10], index=2)
    max_frames = st.number_input("Max frames (0=all)", min_value=0, value=300, step=100)
    max_age = st.slider("Track timeout (frames)", 5, 60, 20, 5)

    run = st.button("▶ Run analysis", type="primary", disabled=(uploaded is None), use_container_width=True)

with live:
    st.markdown('<div class="card"><b>Live Preview</b><div class="small">Yellow = pending • Green = counted • mph shown if calibrated</div></div>', unsafe_allow_html=True)
    live_img_slot = st.empty()
    st.write("")
    progress_bar = st.progress(0.0)

with metrics:
    st.markdown('<div class="card"><b>Status</b><div class="small">Concise + presentable</div></div>', unsafe_allow_html=True)
    live_metrics_slot = st.container()
    st.write("")
    st.markdown('<div class="card"><b>Downloads</b><div class="small">Annotated video + concise CSV + summary JSON</div></div>', unsafe_allow_html=True)
    downloads_slot = st.container()

st.write("")
st.markdown("—")
st.markdown("## Results")

if run and uploaded is not None:
    workdir = tempfile.mkdtemp(prefix="car_alt_calib_")
    in_path = os.path.join(workdir, uploaded.name)
    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Build homography AFTER downscale (points must match resized frame)
    H = None
    if use_calib:
        px_points = [(int(p1x * downscale), int(p1y * downscale)),
                     (int(p2x * downscale), int(p2y * downscale)),
                     (int(p3x * downscale), int(p3y * downscale)),
                     (int(p4x * downscale), int(p4y * downscale))]
        H = build_homography(px_points, float(rect_w), float(rect_l))

    tracker_cfg = {
        "max_age": int(max_age),
        "iou_thr": 0.25,
        "max_dist_px": 80.0,
        "mph_clip": 120.0
    }

    with st.spinner("Running…"):
        annotated, crossings_csv, summary_json, crossings_df, summary = analyze(
            video_path=in_path, outdir=workdir,
            model_name=model_name, conf=conf, iou=iou, imgsz=imgsz,
            max_frames=int(max_frames), downscale=float(downscale), preview_every_n=int(preview_every_n),
            line_orientation=line_orientation, line_pos_frac=float(line_pos_frac), forward_direction=forward_direction,
            H=H, speed_window=int(speed_window) if use_calib else 12,
            tracker_cfg=tracker_cfg,
            live_img_slot=live_img_slot, live_metrics_slot=live_metrics_slot, progress_bar=progress_bar
        )

    # Presentable + concise results
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Forward crossings", summary["forward_crossings"])
    r2.metric("Events logged", summary["events_rows"])
    r3.metric("Elapsed (s)", f"{summary['elapsed_s']:.1f}")
    r4.metric("Speed", "Calibrated" if summary["speed_calibrated"] else "Off")

    st.write("")
    st.video(annotated)

    st.write("")
    st.markdown("### Concise event log (crossings only)")
    st.dataframe(crossings_df.tail(30), use_container_width=True, height=260)

    # downloads
    with downloads_slot:
        with open(annotated, "rb") as f:
            st.download_button("Download annotated_output.mp4", f, file_name="annotated_output.mp4", use_container_width=True)
        with open(crossings_csv, "rb") as f:
            st.download_button("Download crossings.csv", f, file_name="crossings.csv", use_container_width=True)
        with open(summary_json, "rb") as f:
            st.download_button("Download summary.json", f, file_name="summary.json", use_container_width=True)

else:
    st.info("Upload a video → (optional) set 4-point calibration → Run. Green boxes = counted. mph shows only if calibrated.")
