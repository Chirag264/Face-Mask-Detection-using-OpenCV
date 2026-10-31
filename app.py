import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import io
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  CUSTOM CSS — dark glassmorphism theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%); min-height:100vh; }
.hero-header { text-align:center; padding:2.5rem 1rem 1.5rem; }
.hero-header h1 { font-size:3rem; font-weight:800;
    background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:.4rem; }
.hero-header p { color:#94a3b8; font-size:1.1rem; margin:0; }
.glass-card { background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.12);
    border-radius:16px; padding:1.5rem; backdrop-filter:blur(12px); margin-bottom:1.2rem; }
.metric-pill { display:inline-block; padding:6px 16px; border-radius:999px;
    font-size:.85rem; font-weight:600; margin:4px; }
.pill-mask   { background:rgba(52,211,153,.2);  color:#34d399; border:1px solid #34d399; }
.pill-nomask { background:rgba(239,68,68,.2);   color:#f87171; border:1px solid #f87171; }
.pill-info   { background:rgba(96,165,250,.2);  color:#60a5fa; border:1px solid #60a5fa; }
.result-mask   { background:linear-gradient(90deg,#052e16,#14532d); border:1px solid #22c55e;
    border-radius:12px; padding:1rem 1.5rem; color:#4ade80; font-size:1.2rem;
    font-weight:700; text-align:center; margin-top:.8rem; }
.result-nomask { background:linear-gradient(90deg,#450a0a,#7f1d1d); border:1px solid #ef4444;
    border-radius:12px; padding:1rem 1.5rem; color:#f87171; font-size:1.2rem;
    font-weight:700; text-align:center; margin-top:.8rem; }
section[data-testid="stSidebar"] { background:rgba(15,12,41,.85)!important;
    border-right:1px solid rgba(255,255,255,.08); }
hr { border-color:rgba(255,255,255,.1); }
.stButton>button { background:linear-gradient(135deg,#a78bfa,#6366f1); color:white;
    border:none; border-radius:10px; font-weight:600; padding:.6rem 2rem; transition:all .3s; }
.stButton>button:hover { transform:translateY(-2px); box-shadow:0 8px 25px rgba(167,139,250,.4); }
.section-label { color:#a78bfa; font-size:.75rem; font-weight:700;
    text-transform:uppercase; letter-spacing:1.5px; margin-bottom:.5rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")

# ──────────────────────────────────────────────
#  CACHED MODEL LOADING
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_mask_model():
    return load_model(os.path.join(MODEL_DIR, "maskdetect_mobilenet_v2_predictor.model"))

@st.cache_resource(show_spinner=False)
def load_face_detector():
    return cv2.dnn.readNet(
        os.path.join(MODEL_DIR, "deploy.prototxt"),
        os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel"),
    )

# ──────────────────────────────────────────────
#  INFERENCE HELPERS
# ──────────────────────────────────────────────
def detect_faces(image_bgr, face_net, conf_thresh=0.5):
    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300,300), (104.0,177.0,123.0))
    face_net.setInput(blob)
    dets = face_net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        if float(dets[0,0,i,2]) > conf_thresh:
            b = (dets[0,0,i,3:7] * np.array([w,h,w,h])).astype("int")
            boxes.append((max(0,b[0]), max(0,b[1]), min(w-1,b[2]), min(h-1,b[3])))
    return boxes

def predict_masks(image_bgr, boxes, mask_model):
    out = image_bgr.copy()
    results = []
    if not boxes:
        return out, results
    faces = []
    for (sX,sY,eX,eY) in boxes:
        face = image_bgr[sY:eY, sX:eX]
        if face.size == 0:
            continue
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224,224))
        face = preprocess_input(img_to_array(face))
        faces.append(face)
    if faces:
        preds = mask_model.predict(np.array(faces, dtype="float32"), batch_size=16, verbose=0)
        for box, pred in zip(boxes, preds):
            sX,sY,eX,eY = box
            mp, nmp = pred
            label = "Mask" if mp > nmp else "No Mask"
            color = (0,255,0) if label=="Mask" else (0,0,255)
            conf  = max(mp,nmp)*100
            cv2.rectangle(out,(sX,sY),(eX,eY),color,2)
            cv2.putText(out,f"{label}: {conf:.1f}%",(sX,sY-10),
                        cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
            results.append({"label":label,"mask_prob":float(mp),"no_mask_prob":float(nmp),"confidence":float(conf)})
    return out, results

# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    mode = st.radio("Detection Mode", ["📷 Upload Image","🎥 Upload Video","📹 Webcam (Live)"])
    st.markdown("---")
    conf_thresh = st.slider("Face Detection Confidence", .1, 1.0, .5, .05)
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""<div class='glass-card' style='padding:1rem'>
      <span class='metric-pill pill-info'>MobileNetV2</span>
      <span class='metric-pill pill-mask'>99% Accuracy</span><br><br>
      <small style='color:#94a3b8'>Face Detector: Caffe SSD<br>Backbone: MobileNetV2<br>Input: 224×224</small>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<small style='color:#64748b'>Built with ❤️ using TensorFlow, OpenCV & Streamlit</small>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  HERO HEADER
# ──────────────────────────────────────────────
st.markdown("""<div class='hero-header'>
    <h1>😷 Face Mask Detector</h1>
    <p>Real-time mask detection powered by MobileNetV2 + Caffe SSD &nbsp;·&nbsp; 99% accuracy</p>
</div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  LOAD MODELS
# ──────────────────────────────────────────────
with st.spinner("🔄 Loading AI models…"):
    try:
        mask_model = load_mask_model()
        face_net   = load_face_detector()
        models_ok  = True
    except Exception as e:
        models_ok = False
        st.error(f"❌ Could not load models: {e}\n\nEnsure the `Model/` directory exists with required files.")

if not models_ok:
    st.stop()

# ══════════════════════════════════════════════
#  MODE: IMAGE
# ══════════════════════════════════════════════
if mode == "📷 Upload Image":
    st.markdown("<div class='section-label'>Image Detection</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg","jpeg","png"])

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        img_arr = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("**🖼️ Original Image**")
            st.image(pil_img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("🔍 Detecting…"):
            boxes = detect_faces(img_bgr, face_net, conf_thresh)
            annotated_bgr, results = predict_masks(img_bgr, boxes, mask_model)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("**🔎 Detection Result**")
            st.image(annotated_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='section-label'>Detection Summary</div>", unsafe_allow_html=True)

        if not results:
            st.warning("⚠️ No faces detected. Try lowering the confidence threshold.")
        else:
            n_mask    = sum(1 for r in results if r["label"]=="Mask")
            n_no_mask = len(results) - n_mask
            c1,c2,c3 = st.columns(3)
            c1.metric("👤 Faces Detected", len(results))
            c2.metric("✅ With Mask",   n_mask)
            c3.metric("❌ Without Mask", n_no_mask)
            for idx, r in enumerate(results, 1):
                css = "result-mask" if r["label"]=="Mask" else "result-nomask"
                icon = "✅" if r["label"]=="Mask" else "❌"
                st.markdown(f"<div class='{css}'>{icon} Face {idx}: {r['label']} — {r['confidence']:.1f}% confidence</div>", unsafe_allow_html=True)

        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        st.download_button("⬇️ Download Annotated Image", buf.getvalue(), "result.png", "image/png")

# ══════════════════════════════════════════════
#  MODE: VIDEO
# ══════════════════════════════════════════════
elif mode == "🎥 Upload Video":
    st.markdown("<div class='section-label'>Video Detection</div>", unsafe_allow_html=True)
    video_file = st.file_uploader("Upload a video (MP4 / AVI / MOV)", type=["mp4","avi","mov"])

    if video_file:
        st.video(video_file)
        if st.button("🚀 Run Detection on Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(video_file.read())
                tmp_path = tmp_in.name

            cap    = cv2.VideoCapture(tmp_path)
            fps    = cap.get(cv2.CAP_PROP_FPS) or 25
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out_path = tmp_path.replace(".mp4","_out.mp4")
            writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))

            pbar  = st.progress(0, text="Processing…")
            ph    = st.empty()
            stats = {"mask":0,"no_mask":0,"frames":0}
            fi    = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                boxes = detect_faces(frame, face_net, conf_thresh)
                ann, fr = predict_masks(frame, boxes, mask_model)
                writer.write(ann)
                for r in fr:
                    stats["mask" if r["label"]=="Mask" else "no_mask"] += 1
                stats["frames"] += 1
                if fi % 10 == 0:
                    ph.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB), caption=f"Frame {fi}", use_container_width=True)
                pbar.progress(min(int(fi/max(total,1)*100),100), text=f"Frame {fi}/{total}")
                fi += 1

            cap.release(); writer.release()
            pbar.progress(100, text="✅ Done!")
            c1,c2,c3 = st.columns(3)
            c1.metric("🎞️ Frames", stats["frames"])
            c2.metric("✅ Mask Detections", stats["mask"])
            c3.metric("❌ No-Mask Detections", stats["no_mask"])
            with open(out_path,"rb") as f:
                st.download_button("⬇️ Download Annotated Video", f.read(), "result.mp4", "video/mp4")
            os.unlink(tmp_path); os.unlink(out_path)

# ══════════════════════════════════════════════
#  MODE: WEBCAM
# ══════════════════════════════════════════════
elif mode == "📹 Webcam (Live)":
    st.markdown("<div class='section-label'>Webcam Detection</div>", unsafe_allow_html=True)

    cam_mode = st.radio(
        "Camera Input Source",
        ["📸 In-Browser Camera (Works on Cloud & Local)", "🖥️ Native OpenCV Window (Local machine only)"],
        horizontal=True,
    )

    if cam_mode == "📸 In-Browser Camera (Works on Cloud & Local)":
        st.markdown("""<div class='glass-card'>
            <p style='color:#94a3b8;margin:0'>
            Capture a live picture from your device camera to check mask compliance instantly.
            </p></div>""", unsafe_allow_html=True)
        cam_photo = st.camera_input("Take a snapshot")
        if cam_photo:
            pil_img = Image.open(cam_photo).convert("RGB")
            img_arr = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

            with st.spinner("🔍 Detecting…"):
                boxes = detect_faces(img_bgr, face_net, conf_thresh)
                annotated_bgr, results = predict_masks(img_bgr, boxes, mask_model)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("<div class='glass-card'>**📸 Captured Snapshot**", unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='glass-card'>**🔎 Detection Result**", unsafe_allow_html=True)
                st.image(annotated_rgb, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            if not results:
                st.warning("⚠️ No faces detected. Ensure your face is clearly visible.")
            else:
                n_mask = sum(1 for r in results if r["label"] == "Mask")
                n_no_mask = len(results) - n_mask
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("👤 Faces", len(results))
                sc2.metric("✅ With Mask", n_mask)
                sc3.metric("❌ Without Mask", n_no_mask)
                for idx, r in enumerate(results, 1):
                    css = "result-mask" if r["label"] == "Mask" else "result-nomask"
                    icon = "✅" if r["label"] == "Mask" else "❌"
                    st.markdown(f"<div class='{css}'>{icon} Face {idx}: {r['label']} — {r['confidence']:.1f}% confidence</div>", unsafe_allow_html=True)

    else:
        st.markdown("""<div class='glass-card'>
            <p style='color:#94a3b8;margin:0'>
            📌 <strong style='color:#e2e8f0'>Native OpenCV Window</strong><br>
            Runs on your local machine only. Click <em>Start Webcam</em> and press <strong>Q</strong> in the window to quit.
            </p></div>""", unsafe_allow_html=True)

        if st.button("📹 Start Native Webcam"):
            try:
                import imutils
                from imutils.video import VideoStream
                import time
                st.info("🔴 Webcam running — press **Q** in the OpenCV window to quit.")
                vs = VideoStream(src=0).start()
                time.sleep(1.5)
                while True:
                    frame = vs.read()
                    if frame is None:
                        break
                    frame = imutils.resize(frame, width=700)
                    boxes = detect_faces(frame, face_net, conf_thresh)
                    ann, _ = predict_masks(frame, boxes, mask_model)
                    cv2.imshow("Face Mask Detector — Q to quit", ann)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                vs.stop(); cv2.destroyAllWindows()
                st.success("✅ Webcam session ended.")
            except Exception as e:
                st.error(f"Cannot open native display window in this environment: {e}")

# ──────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style='text-align:center;color:#475569;font-size:.85rem;padding-bottom:1.5rem'>
    Face Mask Detection using OpenCV &nbsp;|&nbsp; MobileNetV2 + Caffe SSD &nbsp;|&nbsp;
    <a href='https://github.com' style='color:#a78bfa;text-decoration:none'>GitHub</a>
</div>""", unsafe_allow_html=True)
