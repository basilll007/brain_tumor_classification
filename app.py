import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import tempfile, os, json
from groq import Groq
from dotenv import load_dotenv

# --- ENV & API KEY ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# --- CONFIG ---
MODEL_PATH = "yolov8-brain-tumor.pt"

# --- STREAMLIT SETTINGS ---
st.set_page_config(page_title="Brain Tumor Detection + LLM Report", layout="centered")
st.title("🧠 Brain Tumor Detection + Medical AI Report")

st.markdown("""
Upload a brain scan image (JPG/PNG). YOLOv8 will detect tumor regions.
Groq's `gemma2-9b-it` LLM will then generate a concise medical report.
""")

# --- Load YOLOv8 Model ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# --- Upload Image ---
uploaded = st.file_uploader("📤 Upload Brain Scan", type=["jpg", "jpeg", "png"])

if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(uploaded.read())
    tmp_path = tmp.name

    st.image(tmp_path, caption="Uploaded Image", use_container_width=True)

    # Run detection with image saved
    with st.spinner("Detecting tumors..."):
        results = model.predict(source=tmp_path, conf=0.05, save=True)

    img = Image.open(tmp_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    detections = results[0].boxes
    detections_data = []

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        draw.rectangle([x1, y1, x2, y2], fill=(173, 216, 230, 80), outline=(0, 0, 255), width=2)
        draw.text((x1 + 3, y1 + 3), "Tumor", fill=(0, 0, 255), font=font)
        detections_data.append({"label": "tumor", "bbox": [x1, y1, x2, y2]})

    final_img = Image.alpha_composite(img, overlay).convert("RGB")
    result_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    final_img.save(result_temp_path)

    st.subheader("🖼️ Detection Result")
    st.image(result_temp_path, use_container_width=True)
    st.download_button("📥 Download Annotated Image", open(result_temp_path, "rb"), file_name="tumor_result.jpg")

    # --- Groq LLM Analysis ---
    def analyze_with_groq(detections: list) -> str:
        logs = ""
        for i, d in enumerate(detections):
            logs += f"- Detection {i+1}: Bounding box {d['bbox']}.\n"

        prompt = f"""
You are a medical AI assistant specializing in radiology.

Given the following tumor detection logs, write a clinical-style summary with:
- No patient/date fields
- Do not include confidence scores
- Say "Confidently indicates a possible tumor" for each detection
- Report severity as one word: low, moderate, or high
- End with concise, professional recommendations

Tumor Detection Logs:
{logs}

Format response as a diagnostic report. Avoid any disclaimers.
"""

        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "user", "content": prompt.strip()}
            ]
        )
        return response.choices[0].message.content

    if detections_data:
        st.subheader("📝 AI-Generated Medical Report")
        with st.spinner("Analyzing with Groq..."):
            try:
                report = analyze_with_groq(detections_data)
                st.markdown(report)
                st.download_button("📄 Download Report", report, file_name="diagnostic_report.txt")
            except Exception as e:
                st.error(f"Groq API error: {e}")
    else:
        st.info("✅ No tumors detected. No report generated.")
