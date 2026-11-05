import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Weapon Detection Console", page_icon="🛡️", layout="wide")

HERE = Path(__file__).resolve().parent
MODEL_CANDIDATES = [HERE / "best.pt", HERE / "backend" / "best.pt", HERE.parent / "backend" / "best.pt"]
MAX_VIDEO_MB = 25      # free hosting par RAM/disk kam hoti hai
MAX_FRAMES = 450       # is se lambi video ke sirf pehle itne frames process honge


@st.cache_resource
def load_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return YOLO(str(p))
    raise FileNotFoundError("best.pt nahi mili. Repo mein best.pt rakho.")


# ------------------------------------------------------------------ STYLE
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{ --bg:#F3F5F8; --surface:#FFFFFF; --ink:#14213D; --muted:#5B6577; --line:#DDE2EA; --accent:#1F4E8C; }
html, body, [class*="css"], .stApp { font-family:'IBM Plex Sans','Segoe UI',sans-serif; color:var(--ink); }
.stApp { background:var(--bg); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }
.block-container { padding-top:1.6rem; max-width:1280px; }
.topbar{ background:var(--ink); color:#fff; padding:18px 26px; border-radius:10px; margin-bottom:22px; }
.topbar h1{ font-size:1.35rem; font-weight:600; margin:0; color:#fff; }
.topbar p{ margin:2px 0 0; font-size:.85rem; color:#B8C2D6; }
section[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--line); }
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ font-weight:500; color:var(--muted); padding:10px 18px; }
.stTabs [aria-selected="true"]{ color:var(--accent); }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--accent); }
.panel-title{ font-size:.95rem; font-weight:600; margin:0 0 10px; }
[data-testid="stImage"] img, video{ border-radius:8px; border:1px solid var(--line); max-width:100%; }
.banner{ border-radius:8px; padding:14px 18px; margin:14px 0; border:1px solid; }
.banner b{ font-size:1.05rem; } .banner span{ display:block; font-size:.85rem; margin-top:2px; }
.banner.alert{ background:#FBECEA; border-color:#F0BDB8; color:#8E2119; }
.banner.clear{ background:#E8F4EE; border-color:#B7DDC9; color:#1F5C41; }
[data-testid="stMetric"]{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:14px 16px; }
.stButton>button, .stDownloadButton>button{ border-radius:8px; font-weight:500; border:1px solid var(--line); }
.stButton>button[kind="primary"]{ background:var(--accent); border-color:var(--accent); color:#fff; }
[data-testid="stFileUploader"] section{ background:var(--surface); border:1.5px dashed #B9C3D3; border-radius:10px; }
.hint{ color:var(--muted); font-size:.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


def banner(count: int, unit: str = "object detected"):
    if count:
        st.markdown(
            f'<div class="banner alert"><b>{count} {unit}</b><span>Result mein boxes check karo.</span></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="banner clear"><b>Koi object nahi mila</b>'
            "<span>Confidence slider kam karke dobara try karo.</span></div>",
            unsafe_allow_html=True)


st.markdown(
    '<div class="topbar"><h1>Weapon Detection Console</h1>'
    "<p>YOLO11 based image and video analysis</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Kam value: zyada detections, lekin galat bhi ho sakti hain.")
    st.markdown('<p class="hint">Model kamzor ho to 0.10 se 0.20 try karo.</p>', unsafe_allow_html=True)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

tab_img, tab_vid = st.tabs(["Image", "Video"])

# ------------------------------------------------------------------ IMAGE
with tab_img:
    up = st.file_uploader("Image upload karo (JPG, PNG, JFIF)",
                          type=["jpg", "jpeg", "png", "jfif"], key="img")
    if up is not None:
        img = Image.open(up).convert("RGB")
        with st.spinner("Analyzing..."):
            result = model.predict(img, conf=conf, verbose=False)[0]
        annotated = Image.fromarray(result.plot()[..., ::-1])

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<p class="panel-title">Original</p>', unsafe_allow_html=True)
            st.image(img)
        with right:
            st.markdown('<p class="panel-title">Detection result</p>', unsafe_allow_html=True)
            st.image(annotated)

        dets = [{"Label": model.names[int(b.cls)],
                 "Confidence": round(float(b.conf), 3),
                 "Box (x1,y1,x2,y2)": [round(x, 1) for x in b.xyxy[0].tolist()]}
                for b in result.boxes]
        banner(len(dets))
        m1, m2, m3 = st.columns(3)
        m1.metric("Objects", len(dets))
        m2.metric("Highest confidence", f"{max(d['Confidence'] for d in dets):.0%}" if dets else "-")
        m3.metric("Threshold", f"{conf:.0%}")
        if dets:
            st.dataframe(dets, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ VIDEO
with tab_vid:
    vid = st.file_uploader("Video upload karo (MP4, MOV, AVI, MKV)",
                           type=["mp4", "mov", "avi", "mkv"], key="vid")
    st.markdown(f'<p class="hint">Limit: {MAX_VIDEO_MB} MB, pehle {MAX_FRAMES} frames process hote hain.</p>',
                unsafe_allow_html=True)

    if vid is not None:
        if vid.size > MAX_VIDEO_MB * 1024 * 1024:
            st.error(f"Video {MAX_VIDEO_MB} MB se bari hai. Chhoti video upload karo.")
        else:
            left, right = st.columns(2, gap="large")
            with left:
                st.markdown('<p class="panel-title">Original video</p>', unsafe_allow_html=True)
                st.video(vid.getvalue())

            with right:
                st.markdown('<p class="panel-title">Detection result</p>', unsafe_allow_html=True)
                if st.button("Analyze video", type="primary"):
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp = Path(tmp)
                        src = tmp / "in.mp4"
                        src.write_bytes(vid.getvalue())
                        cap = cv2.VideoCapture(str(src))
                        if not cap.isOpened():
                            st.error("Video open nahi hui.")
                        else:
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            total_est = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_FRAMES, MAX_FRAMES)
                            raw = tmp / "raw.mp4"
                            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                            bar = st.progress(0.0, text="Processing...")
                            total = hit = 0
                            while total < MAX_FRAMES:
                                ok, frame = cap.read()
                                if not ok:
                                    break
                                res = model.predict(frame, conf=conf, verbose=False)[0]
                                total += 1
                                hit += 1 if len(res.boxes) else 0
                                writer.write(res.plot())
                                bar.progress(min(total / max(total_est, 1), 1.0),
                                             text=f"Frame {total} / {total_est}")
                            cap.release()
                            writer.release()
                            bar.empty()

                            out = tmp / "out.mp4"
                            subprocess.run(
                                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw),
                                 "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out)],
                                check=True, capture_output=True)
                            st.session_state["vid_result"] = {
                                "bytes": out.read_bytes(), "hit": hit, "total": total}

                vr = st.session_state.get("vid_result")
                if vr:
                    st.video(vr["bytes"])
                    st.download_button("Download result video", vr["bytes"], "result.mp4", "video/mp4")

            vr = st.session_state.get("vid_result")
            if vr:
                banner(1 if vr["hit"] else 0, "or zyada frames mein object mila")
                m1, m2, m3 = st.columns(3)
                m1.metric("Frames processed", vr["total"])
                m2.metric("Frames with detection", vr["hit"])
                m3.metric("Detection rate", f"{(vr['hit'] / vr['total']) if vr['total'] else 0:.0%}")
