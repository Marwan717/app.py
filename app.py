# app.py — Smarter Layout + Better Counting (Line-Crossing) + Specific Classes
# Cloud-safe: no lap/bytetrack. Uses YOLO detect + lightweight tracker.
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

KEEP = {0:"person", 1:"bicycle", 2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}
ORDER = ["person","bicycle","motorcycle","car","bus","truck"]
NICE = {"person":"Pedestrians","bicycle":"Bicycles","motorcycle":"Motorcycles","car":"Cars","bus":"Buses","truck":"Trucks"}

YELLOW = (0,255,255)  # boxes
CYAN = (255,255,0)    # line


def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w,h))


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


class SpeedProjector:
    def __init__(self, meters_per_pixel=None):
        self.mpp = meters_per_pixel

    def project(self, cx, cy):
        if self.mpp and self.mpp > 0:
            return float(cx*self.mpp), float(cy*self.mpp)
        return None


def point_side_of_line(px, py, x1, y1, x2, y2):
    # sign of cross product: positive one side, negative other side
    return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)


class SimpleTracker:
    """
    Lightweight tracker that is 'good enough' for line-crossing counts:
    - Greedy IoU then centroid distance matching per class
    - Stores last side-of-line state to count crossings reliably
    """
    def __init__(self, fps, projector, max_age=20, iou_thr=0.25, max_dist_px=80.0,
                 speed_window=12, speed_clip_mph=120.0):
        self.fps = fps
        self.projector = projector
        self.max_age = max_age
        self.iou_thr = iou_thr
        self.max_dist = max_dist_px
        self.speed_window = int(max(3, speed_window))
        self.speed_clip = float(speed_clip_mph)

        self.next_id = 1
        self.tracks = {}  # id-> dict

    def _new_track(self, det):
        tid = self.next_id; self.next_id += 1
        self.tracks[tid] = {
            "id": tid,
            "class": det["class"],
            "bbox": det["bbox"],
            "cx": det["cx"], "cy": det["cy"],
            "last_frame": det["frame"],
            "hist": deque(maxlen=60),  # (frame, gx, gy)
            "speed_mph": None,
            "line_side": None,
            "counted": False,  # used per line direction counting
        }
        gp = self.projector.project(det["cx"], det["cy"])
        if gp is not None:
            self.tracks[tid]["hist"].append((det["frame"], gp[0], gp[1]))
        return tid

    def _speed_mph(self, tr):
        if len(tr["hist"]) < 2:
            return None
        hist = list(tr["hist"])[-self.speed_window:]
        if len(hist) < 2:
            return None
        f0,x0,y0 = hist[0]
        f1,x1,y1 = hist[-1]
        dt = (f1-f0)/self.fps
        if dt <= 1e-6:
            return None
        dist_m = float(np.hypot(x1-x0, y1-y0))
        mph = (dist_m/dt)*2.2369362921
        if mph < 0 or mph > self.speed_clip:
            return None
        return mph

    def update(self, dets, frame_idx):
        # age out
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
                tr = self.tracks[tid]
                if tr["class"] != d["class"]:
                    continue
                iou = iou_xyxy(tr["bbox"], d["bbox"])
                if iou >= self.iou_thr:
                    cand.append((iou, tid, di))
        cand.sort(reverse=True, key=lambda x: x[0])

        for _, tid, di in cand:
            if di not in unmatched or tid in used:
                continue
            used.add(tid); unmatched.remove(di)
            matches.append((tid,di))

        # centroid match
        for di in list(unmatched):
            d = dets[di]
            best = None
            for tid in track_ids:
                if tid in used:
                    continue
                tr = self.tracks[tid]
                if tr["class"] != d["class"]:
                    continue
                dist = float(np.hypot(d["cx"]-tr["cx"], d["cy"]-tr["cy"]))
                if dist <= self.max_dist and (best is None or dist < best[0]):
                    best = (dist, tid)
            if best is not None:
                _, tid = best
                used.add(tid); unmatched.remove(di)
                matches.append((tid,di))

        # update matches
        for tid, di in matches:
            d = dets[di]
            tr = self.tracks[tid]
            tr["bbox"] = d["bbox"]
            tr["cx"], tr["cy"] = d["cx"], d["cy"]
            tr["last_frame"] = d["frame"]

            gp = self.projector.project(d["cx"], d["cy"])
            if gp is not None:
                tr["hist"].append((d["frame"], gp[0], gp[1]))
                tr["speed_mph"] = self._speed_mph(tr)
            else:
                tr["speed_mph"] = None

            d["track_id"] = tid
            d["speed_mph"] = tr["speed_mph"]

        # new tracks
        for di in list(unmatched):
            tid = self._new_track(dets[di])
            dets[di]["track_id"] = tid
            dets[di]["speed_mph"] = self.tracks[tid]["speed_mph"]

        return dets


def analyze(video_path, outdir, model_name, conf, iou, imgsz,
            max_frames, downscale, preview_every_n,
            mpp, line_y_frac, line_dir,  # line counting
            tracker_cfg, live_img, live_panel):
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

    projector = SpeedProjector(meters_per_pixel=mpp)
    tracker = SimpleTracker(
        fps=fps,
        projector=projector,
        max_age=tracker_cfg["max_age"],
        iou_thr=tracker_cfg["iou_thr"],
        max_dist_px=tracker_cfg["max_dist_px"] * downscale,
        speed_window=tracker_cfg["speed_window"],
        speed_clip_mph=tracker_cfg["speed_clip_mph"],
    )

    # Counting line: horizontal across frame at y = frac*h
    y_line = int(line_y_frac * h)
    x1,y1,x2,y2 = 0,y_line,w-1,y_line

    # flow counts by class
    flow_counts = defaultdict(int)

    # approx unique by IDs (still unstable)
    approx_unique_ids = defaultdict(set)

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
                dets.append({"frame": fidx, "class": cname, "conf": float(cf),
                             "bbox": bb.astype(float), "cx": float(cx), "cy": float(cy)})

        dets = tracker.update(dets, fidx)

        # draw line
        cv2.line(frame, (x1,y1), (x2,y2), CYAN, 3)
        cv2.putText(frame, "COUNT LINE", (10, y_line - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)

        # update flow counts using line-side sign flip
        for d in dets:
            cname = d["class"]
            tid = int(d["track_id"])
            mph = d.get("speed_mph")

            approx_unique_ids[cname].add(tid)

            # determine side of line
            side = point_side_of_line(d["cx"], d["cy"], x1,y1,x2,y2)

            tr = tracker.tracks.get(tid)
            if tr is not None:
                prev_side = tr["line_side"]
                tr["line_side"] = side

                # crossing detected when sign changes from + to - or - to +
                if prev_side is not None and (prev_side == 0 or side == 0 or (prev_side > 0) != (side > 0)):
                    # apply direction rule:
                    # "down" means crossing from above->below (prev negative? depends on sign)
                    # For a horizontal line, sign is proportional to (py - y_line).
                    # If py > y_line => side positive; if py < y_line => side negative.
                    moved_down = (tr["cy"] > y_line)  # after update
                    moved_up = (tr["cy"] < y_line)

                    if line_dir == "Both":
                        flow_counts[cname] += 1
                        tr["counted"] = True
                    elif line_dir == "Down only" and moved_down:
                        flow_counts[cname] += 1
                        tr["counted"] = True
                    elif line_dir == "Up only" and moved_up:
                        flow_counts[cname] += 1
                        tr["counted"] = True

            # draw box + label
            x1b,y1b,x2b,y2b = map(int, d["bbox"].tolist())
            cv2.rectangle(frame, (x1b,y1b), (x2b,y2b), YELLOW, 3)

            if mph is not None:
                label = f"{cname}  #{tid}  {mph:.1f} mph"
            else:
                label = f"{cname}  #{tid}"

            cv2.putText(frame, label, (x1b, max(24, y1b-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

            rows.append({
                "frame": fidx, "track_id": tid, "class": cname, "conf": d["conf"],
                "cx": d["cx"], "cy": d["cy"],
                "x1": float(d["bbox"][0]), "y1": float(d["bbox"][1]), "x2": float(d["bbox"][2]), "y2": float(d["bbox"][3]),
                "speed_mph": float(mph) if mph is not None else None
            })

        cv2.putText(frame, f"Frame {fidx}/{total_frames if total_frames else '?'}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 2)

        writer.write(frame)

        if preview_every_n <= 1 or fidx % preview_every_n == 0:
            live_img.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # panel dashboard
            live_panel.markdown("### Live Panel")
            cols = live_panel.columns(2)
            cols[0].metric("Frame", fidx)
            cols[1].metric("Flow total", sum(flow_counts.values()))

            live_panel.write("**Flow counts (most reliable)**")
            live_panel.json({k: flow_counts.get(k, 0) for k in ORDER})

            live_panel.write("**This frame detections**")
            live_panel.json({k: frame_counts.get(k, 0) for k in ORDER})

            live_panel.write("**Approx unique IDs (unstable)**")
            live_panel.json({k: len(approx_unique_ids.get(k, set())) for k in ORDER})

        fidx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "tracks.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "flow_counts": {k: int(flow_counts.get(k, 0)) for k in ORDER},
        "approx_unique_ids": {k: int(len(approx_unique_ids.get(k, set()))) for k in ORDER},
        "processed_frames": int(fidx),
        "elapsed_s": float(elapsed),
        "meters_per_pixel": mpp,
        "line_y_frac": float(line_y_frac),
        "line_direction": line_dir,
    }
    summary_json = os.path.join(outdir, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    return annotated, csv_path, summary_json, df, summary


# --------------------------- UI ---------------------------

st.set_page_config(page_title="Traffic Analytics Dashboard", layout="wide")

st.markdown("""
<style>
.kpi {padding: 14px 16px; border-radius: 16px; background: #0b1220; border: 1px solid rgba(255,255,255,0.08);}
.kpi h3 {margin: 0; font-size: 14px; opacity: 0.85;}
.kpi p {margin: 6px 0 0 0; font-size: 26px; font-weight: 700;}
.muted {opacity: 0.75;}
</style>
""", unsafe_allow_html=True)

st.title("🚦 Traffic Video Analytics Dashboard")
st.write("More specific counts (cars/buses/trucks/pedestrians/bikes) + **smarter counting** using a **line-crossing flow counter**.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    conf = st.slider("Confidence", 0.05, 0.80, 0.20, 0.05)
    iou = st.slider("IoU", 0.10, 0.90, 0.50, 0.05)
    imgsz = st.select_slider("Image size", [640, 768, 896, 960, 1024], value=768)

    st.divider()
    st.subheader("Speed (mph) calibration")
    st.caption("If 0, mph will be blank. Use a reasonable m/px estimate.")
    mpp = st.number_input("Meters per pixel (m/px)", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
    mpp = mpp if mpp > 0 else None

    st.divider()
    st.subheader("Counting (recommended)")
    line_y_frac = st.slider("Count line position (0=top, 1=bottom)", 0.1, 0.9, 0.55, 0.01)
    line_dir = st.selectbox("Count direction", ["Both", "Down only", "Up only"], index=0)

    st.divider()
    st.subheader("Performance")
    downscale = st.selectbox("Downscale", [1.0, 0.75, 0.5], index=1)
    preview_every_n = st.selectbox("Preview every N frames", [1,2,3,5,10], index=2)
    max_frames = st.number_input("Max frames (0=all)", min_value=0, value=300, step=100)

    st.divider()
    st.subheader("Tracker tuning")
    speed_window = st.slider("Speed smoothing (frames)", 5, 45, 12, 5)
    max_age = st.slider("Track timeout (frames)", 5, 60, 20, 5)

uploaded = st.file_uploader("Drop your video here", type=["mp4","mov","avi","mkv"])
run = st.button("Run analysis", type="primary", disabled=(uploaded is None))

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "⚙️ How the counts work", "⬇️ Downloads"])

with tab1:
    # KPI row placeholders
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_slots = [k1.empty(), k2.empty(), k3.empty(), k4.empty(), k5.empty()]

    st.divider()
    left, right = st.columns([2,1])
    live_img = left.empty()
    live_panel = right.empty()

    st.divider()
    final = st.container()

    if run and uploaded is not None:
        workdir = tempfile.mkdtemp(prefix="traffic_dash_")
        in_path = os.path.join(workdir, uploaded.name)
        with open(in_path, "wb") as f:
            f.write(uploaded.getbuffer())

        tracker_cfg = {"max_age": int(max_age), "iou_thr": 0.25, "max_dist_px": 80.0,
                       "speed_window": int(speed_window), "speed_clip_mph": 120.0}

        st.success("Processing… Live preview + flow counts updating.")
        annotated, csv_path, summary_json, df, summary = analyze(
            video_path=in_path,
            outdir=workdir,
            model_name=model_name,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_frames=int(max_frames),
            downscale=float(downscale),
            preview_every_n=int(preview_every_n),
            mpp=mpp,
            line_y_frac=float(line_y_frac),
            line_dir=line_dir,
            tracker_cfg=tracker_cfg,
            live_img=live_img,
            live_panel=live_panel
        )

        # KPI cards
        flow_total = sum(summary["flow_counts"].values())
        approx_unique_total = sum(summary["approx_unique_ids"].values())

        labels = ["Frames", "Elapsed (s)", "Flow total", "Cars flow", "Trucks flow"]
        values = [
            summary["processed_frames"],
            f"{summary['elapsed_s']:.1f}",
            flow_total,
            summary["flow_counts"].get("car", 0),
            summary["flow_counts"].get("truck", 0),
        ]

        for slot, lab, val in zip(kpi_slots, labels, values):
            slot.markdown(f"<div class='kpi'><h3>{lab}</h3><p>{val}</p></div>", unsafe_allow_html=True)

        with final:
            st.subheader("Final Annotated Video")
            st.video(annotated)

            st.subheader("Counts by Class (Flow = best)")
            cA, cB = st.columns(2)

            with cA:
                st.markdown("**Flow counts (line crossing)**")
                st.json({NICE[k]: summary["flow_counts"].get(k, 0) for k in ORDER})
                st.markdown("<div class='muted'>Most reliable for “how many passed”.</div>", unsafe_allow_html=True)

            with cB:
                st.markdown("**Approx unique IDs (unstable)**")
                st.json({NICE[k]: summary["approx_unique_ids"].get(k, 0) for k in ORDER})
                st.markdown("<div class='muted'>Can be wrong if objects overlap or IDs swap.</div>", unsafe_allow_html=True)

            st.subheader("Speed sanity check")
            if mpp and df["speed_mph"].notna().any():
                s = df["speed_mph"].dropna()
                st.write(f"Median mph: **{s.median():.1f}**, 90th percentile: **{s.quantile(0.9):.1f}**")
            else:
                st.info("mph is off unless you set meters-per-pixel.")

            st.subheader("Sample table")
            st.dataframe(df.head(200), use_container_width=True)

            # stash paths in session for downloads tab
            st.session_state["out_annotated"] = annotated
            st.session_state["out_csv"] = csv_path
            st.session_state["out_json"] = summary_json

with tab2:
    st.markdown(
        """
        ## Why “unique IDs” are not accurate
        Without a strong tracker + re-ID, IDs can swap when:
        - vehicles overlap / occlude
        - pedestrians cluster
        - objects exit and re-enter

        ## What you should trust instead
        **Flow counts**:
        - We draw a counting line across the frame
        - An object is counted when its tracked centroid crosses the line
        - This is the standard approach for traffic volume estimation

        ## Best setup
        - Put the line somewhere vehicles clearly cross (not near stop lines where they bounce)
        - Choose direction if you want one-way counts
        """
    )

with tab3:
    st.subheader("Download outputs")
    if "out_annotated" in st.session_state:
        with open(st.session_state["out_annotated"], "rb") as f:
            st.download_button("Download annotated_output.mp4", f, file_name="annotated_output.mp4")
        with open(st.session_state["out_csv"], "rb") as f:
            st.download_button("Download tracks.csv", f, file_name="tracks.csv")
        with open(st.session_state["out_json"], "rb") as f:
            st.download_button("Download summary.json", f, file_name="summary.json")
    else:
        st.info("Run the analysis first to generate downloads.")
