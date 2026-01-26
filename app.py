# app.py — Cars only + forward line-cross count + box turns GREEN after crossing
# Cloud-safe: no lap/bytetrack. YOLO detect + lightweight tracker.
#
# requirements.txt:
# streamlit
# ultralytics
# opencv-python-headless
# numpy
# pandas

import os, json, time, tempfile
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

CAR_CLASS_ID = 2  # COCO car
YELLOW = (0, 255, 255)   # before crossing
GREEN  = (0, 255, 0)     # after crossing
CYAN   = (255, 255, 0)   # line
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
    # sign of cross product
    return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)

class CarTracker:
    """
    Lightweight tracker good enough to:
    - maintain stable-ish IDs
    - detect line crossing
    - mark crossed cars so box becomes GREEN
    """
    def __init__(self, fps, max_age=20, iou_thr=0.25, max_dist_px=80.0):
        self.fps = fps
        self.max_age = max_age
        self.iou_thr = iou_thr
        self.max_dist = max_dist_px
        self.next_id = 1
        self.tracks = {}  # tid -> dict

    def _new_track(self, det):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            "id": tid,
            "bbox": det["bbox"],
            "cx": det["cx"], "cy": det["cy"],
            "last_frame": det["frame"],
            "crossed": False,
            "prev_cx": det["cx"], "prev_cy": det["cy"],
            "prev_side": None,
        }
        return tid

    def update(self, dets, frame_idx):
        # remove dead
        dead = [tid for tid,tr in self.tracks.items() if frame_idx - tr["last_frame"] > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        if not dets:
            return []

        track_ids = list(self.tracks.keys())
        unmatched = set(range(len(dets)))
        used = set()
        matches = []

        # IoU match
        cand = []
        for di,d in enumerate(dets):
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
            used.add(tid); unmatched.remove(di)
            matches.append((tid, di))

        # centroid distance match
        for di in list(unmatched):
            d = dets[di]
            best = None
            for tid in track_ids:
                if tid in used:
                    continue
                tr = self.tracks[tid]
                dist = float(np.hypot(d["cx"]-tr["cx"], d["cy"]-tr["cy"]))
                if dist <= self.max_dist and (best is None or dist < best[0]):
                    best = (dist, tid)
            if best is not None:
                _, tid = best
                used.add(tid); unmatched.remove(di)
                matches.append((tid, di))

        # apply updates
        for tid, di in matches:
            d = dets[di]
            tr = self.tracks[tid]
            tr["bbox"] = d["bbox"]
            tr["prev_cx"], tr["prev_cy"] = tr["cx"], tr["cy"]
            tr["cx"], tr["cy"] = d["cx"], d["cy"]
            tr["last_frame"] = d["frame"]

            d["track_id"] = tid
            d["crossed"] = tr["crossed"]

        # new tracks
        for di in list(unmatched):
            tid = self._new_track(dets[di])
            dets[di]["track_id"] = tid
            dets[di]["crossed"] = self.tracks[tid]["crossed"]

        return dets


def analyze(
    video_path, outdir, model_name, conf, iou, imgsz,
    max_frames, downscale, preview_every_n,
    line_orientation, line_pos_frac, forward_direction,
    tracker_cfg, live_img, live_panel
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    w = int(w0 * downscale); h = int(h0 * downscale)
    if w <= 0 or h <= 0:
        w,h = w0,h0
        downscale = 1.0

    model = YOLO(model_name)

    annotated = os.path.join(outdir, "annotated_output.mp4")
    writer = make_writer(annotated, fps, w, h)

    tracker = CarTracker(
        fps=fps,
        max_age=tracker_cfg["max_age"],
        iou_thr=tracker_cfg["iou_thr"],
        max_dist_px=tracker_cfg["max_dist_px"] * downscale
    )

    # Define counting line
    if line_orientation == "Horizontal":
        y_line = int(line_pos_frac * h)
        x1,y1,x2,y2 = 0,y_line,w-1,y_line
    else:
        x_line = int(line_pos_frac * w)
        x1,y1,x2,y2 = x_line,0,x_line,h-1

    # Count
    flow_count = 0
    rows = []

    t0 = time.time()
    fidx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and fidx >= max_frames:
            break

        if downscale != 1.0:
            frame = cv2.resize(frame, (w,h), interpolation=cv2.INTER_AREA)

        res = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

        dets = []
        # Only CAR detections
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

        # draw counting line
        cv2.line(frame, (x1,y1), (x2,y2), CYAN, 3)
        cv2.putText(frame, "COUNT LINE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2)

        # crossing logic per track
        for d in dets:
            tid = int(d["track_id"])
            tr = tracker.tracks.get(tid)
            if tr is None:
                continue

            # compute side now
            side_now = side_of_line(tr["cx"], tr["cy"], x1,y1,x2,y2)
            side_prev = tr["prev_side"]
            tr["prev_side"] = side_now  # store for next loop

            # Determine forward crossing direction
            crossed_now = False
            if side_prev is not None and (side_prev == 0 or side_now == 0 or (side_prev > 0) != (side_now > 0)):
                # For horizontal line:
                #   above -> below means cy increased past y_line
                # For vertical line:
                #   left  -> right means cx increased past x_line
                if line_orientation == "Horizontal":
                    moved_forward = (tr["prev_cy"] < y_line and tr["cy"] >= y_line) if forward_direction == "Down" \
                                    else (tr["prev_cy"] > y_line and tr["cy"] <= y_line)
                else:
                    moved_forward = (tr["prev_cx"] < x_line and tr["cx"] >= x_line) if forward_direction == "Right" \
                                    else (tr["prev_cx"] > x_line and tr["cx"] <= x_line)

                if moved_forward and not tr["crossed"]:
                    tr["crossed"] = True
                    flow_count += 1
                    crossed_now = True

            # draw box: yellow before crossing, green after crossing
            color = GREEN if tr["crossed"] else YELLOW
            x1b,y1b,x2b,y2b = map(int, tr["bbox"].tolist())
            cv2.rectangle(frame, (x1b,y1b), (x2b,y2b), color, 3)

            label = f"car  #{tid}" + ("  ✅" if tr["crossed"] else "")
            cv2.putText(frame, label, (x1b, max(24, y1b-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            rows.append({
                "frame": fidx,
                "track_id": tid,
                "conf": float(d["conf"]),
                "x1": float(tr["bbox"][0]), "y1": float(tr["bbox"][1]),
                "x2": float(tr["bbox"][2]), "y2": float(tr["bbox"][3]),
                "cx": float(tr["cx"]), "cy": float(tr["cy"]),
                "crossed": bool(tr["crossed"]),
                "crossed_this_frame": bool(crossed_now),
            })

        # HUD
        cv2.putText(frame, f"Cars crossed (forward): {flow_count}",
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        cv2.putText(frame, f"Frame {fidx}/{total_frames if total_frames else '?'}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

        writer.write(frame)

        if preview_every_n <= 1 or fidx % preview_every_n == 0:
            live_img.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            live_panel.markdown("### Live Counts")
            live_panel.metric("Forward car crossings", flow_count)
            live_panel.metric("Cars currently detected", len(dets))

        fidx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "cars_tracks.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "forward_car_crossings": int(flow_count),
        "processed_frames": int(fidx),
        "elapsed_s": float(elapsed),
        "line_orientation": line_orientation,
        "line_pos_frac": float(line_pos_frac),
        "forward_direction": forward_direction
    }
    json_path = os.path.join(outdir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return annotated, csv_path, json_path, df, summary


# --------------------------- Streamlit UI (Smarter Layout) ---------------------------

st.set_page_config(page_title="Cars Forward Count", layout="wide")

st.markdown("""
<style>
.kpi {padding: 14px 16px; border-radius: 16px; background: #0b1220; border: 1px solid rgba(255,255,255,0.08);}
.kpi h3 {margin: 0; font-size: 14px; opacity: 0.85;}
.kpi p {margin: 6px 0 0 0; font-size: 26px; font-weight: 700;}
.muted {opacity: 0.75;}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Car Line-Cross Counter (Forward Only)")
st.write("Only **cars** are detected. When a car crosses the line in the **forward direction**, it is counted and its box turns **GREEN**.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"], index=0)
    conf = st.slider("Confidence", 0.05, 0.80, 0.20, 0.05)
    iou = st.slider("IoU", 0.10, 0.90, 0.50, 0.05)
    imgsz = st.select_slider("Image size", [640, 768, 896, 960], value=768)

    st.divider()
    st.subheader("Counting line")
    line_orientation = st.selectbox("Orientation", ["Horizontal", "Vertical"], index=0)
    line_pos_frac = st.slider("Line position (0=top/left, 1=bottom/right)", 0.1, 0.9, 0.55, 0.01)

    if line_orientation == "Horizontal":
        forward_direction = st.selectbox("Forward direction", ["Down", "Up"], index=0)
    else:
        forward_direction = st.selectbox("Forward direction", ["Right", "Left"], index=0)

    st.divider()
    st.subheader("Performance")
    downscale = st.selectbox("Downscale", [1.0, 0.75, 0.5], index=1)
    preview_every_n = st.selectbox("Preview every N frames", [1,2,3,5,10], index=2)
    max_frames = st.number_input("Max frames (0=all)", min_value=0, value=300, step=100)

    st.divider()
    st.subheader("Tracker")
    max_age = st.slider("Track timeout (frames)", 5, 60, 20, 5)

uploaded = st.file_uploader("Drop your video", type=["mp4","mov","avi","mkv"])
run = st.button("Run", type="primary", disabled=(uploaded is None))

tab1, tab2 = st.tabs(["📊 Dashboard", "⬇️ Downloads"])

with tab1:
    k1, k2, k3, k4 = st.columns(4)
    kpiA, kpiB, kpiC, kpiD = k1.empty(), k2.empty(), k3.empty(), k4.empty()

    st.divider()
    left, right = st.columns([2,1])
    live_img = left.empty()
    live_panel = right.empty()

    st.divider()
    final = st.container()

    if run and uploaded is not None:
        workdir = tempfile.mkdtemp(prefix="car_forward_")
        in_path = os.path.join(workdir, uploaded.name)
        with open(in_path, "wb") as f:
            f.write(uploaded.getbuffer())

        tracker_cfg = {"max_age": int(max_age), "iou_thr": 0.25, "max_dist_px": 80.0}

        st.success("Processing… cars only. Green boxes mean counted.")
        annotated, csv_path, json_path, df, summary = analyze(
            video_path=in_path,
            outdir=workdir,
            model_name=model_name,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_frames=int(max_frames),
            downscale=float(downscale),
            preview_every_n=int(preview_every_n),
            line_orientation=line_orientation,
            line_pos_frac=float(line_pos_frac),
            forward_direction=forward_direction,
            tracker_cfg=tracker_cfg,
            live_img=live_img,
            live_panel=live_panel
        )

        kpiA.markdown(f"<div class='kpi'><h3>Forward crossings</h3><p>{summary['forward_car_crossings']}</p></div>", unsafe_allow_html=True)
        kpiB.markdown(f"<div class='kpi'><h3>Frames</h3><p>{summary['processed_frames']}</p></div>", unsafe_allow_html=True)
        kpiC.markdown(f"<div class='kpi'><h3>Elapsed (s)</h3><p>{summary['elapsed_s']:.1f}</p></div>", unsafe_allow_html=True)
        kpiD.markdown(f"<div class='kpi'><h3>Line</h3><p>{summary['line_orientation']} {summary['forward_direction']}</p></div>", unsafe_allow_html=True)

        with final:
            st.subheader("Final annotated video")
            st.video(annotated)
            st.subheader("Sample rows")
            st.dataframe(df.head(200), use_container_width=True)

            st.session_state["out_annotated"] = annotated
            st.session_state["out_csv"] = csv_path
            st.session_state["out_json"] = json_path

with tab2:
    st.subheader("Downloads")
    if "out_annotated" in st.session_state:
        with open(st.session_state["out_annotated"], "rb") as f:
            st.download_button("Download annotated_output.mp4", f, file_name="annotated_output.mp4")
        with open(st.session_state["out_csv"], "rb") as f:
            st.download_button("Download cars_tracks.csv", f, file_name="cars_tracks.csv")
        with open(st.session_state["out_json"], "rb") as f:
            st.download_button("Download summary.json", f, file_name="summary.json")
    else:
        st.info("Run the analysis first.")
