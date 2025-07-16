import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import tempfile, os, glob, json
from groq import Groq

# --- CONFIG ---
MODEL_PATH = "yolov8-brain-tumor.pt"
GROQ_API_KEY = "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"  # 🔐 Replace with your key
client = Groq(api_key=GROQ_API_KEY)

# --- STREAMLIT SETTINGS ---
st.set_page_config(page_title="Brain Tumor Detection + AI Report", layout="centered")
st.title("🧠 Brain Tumor Detection + LLM Report (Groq)")
st.markdown("""
Upload a brain scan image (.jpg/.png). The model will detect tumor regions,
and a medical-style report will be generated using Groq LLM (`gemma2-9b-it`).
""")

# --- Load YOLOv8 Model ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# --- Upload Image ---
uploaded = st.file_uploader("📤 Upload Scan", type=["jpg", "jpeg", "png"])
if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(uploaded.read())
    tmp_path = tmp.name

    st.image(tmp_path, caption="Uploaded Image", use_container_width=True)

    # --- Run Detection ---
    with st.spinner("Running tumor detection..."):
        results = model.predict(source=tmp_path, conf=0.05)

    img = Image.open(tmp_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    detections = results[0].boxes
    prompts = []

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        draw.rectangle([x1, y1, x2, y2], fill=(173, 216, 230, 80), outline=(0, 0, 255), width=2)
        draw.text((x1+3, y1+3), f"Tumor {conf:.2f}", fill=(0, 0, 255), font=font)
        prompts.append({"label": "tumor", "confidence": conf, "bbox": [x1, y1, x2, y2]})

    final_img = Image.alpha_composite(img, overlay).convert("RGB")

    latest_dir = sorted(glob.glob("runs/detect/predict*"), key=os.path.getmtime)[-1]
    result_path = os.path.join(latest_dir, os.path.basename(tmp_path))
    final_img.save(result_path)

    st.subheader("🖼️ Detection Result")
    st.image(result_path, use_container_width=True)
    st.download_button("📥 Download Annotated Image", open(result_path, "rb"), file_name="annotated.jpg")

    # --- Generate Groq Medical Report ---
    def analyze_with_groq(detections: list) -> str:
        log_text = "Tumor Detection Log:\n"
        for i, d in enumerate(detections):
            log_text += (
                f"- Detection {i+1}:\n"
                f"  Confidence: {d['confidence']:.2f}\n"
                f"  BBox: {d['bbox']}\n"
            )

        prompt = (
            "You're a medical AI assistant. Based on the following detection logs, "
            "analyze the presence and severity of brain tumors and provide diagnostic insights:\n\n"
            f"{log_text}\n\n"
            "Return a medical-style report summarizing findings, severity, and recommendations."
        )

        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    if prompts:
        st.subheader("📝 AI-Generated Medical Report")
        with st.spinner("Analyzing with Groq LLM..."):
            try:
                report = analyze_with_groq(prompts)
                st.markdown(report)
                st.download_button("📄 Download Report", report, file_name="tumor_report.txt")
            except Exception as e:
                st.error(f"Groq LLM error: {e}")
    else:
        st.info("✅ No tumors detected. No report generated.")
