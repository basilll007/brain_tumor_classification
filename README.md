# 🧠 Brain Tumor Detection and Diagnostic Report System

This is a Streamlit-based web application that performs real-time **brain tumor detection** from uploaded medical scan images and generates a **diagnostic report** using a large language model (LLM) via Groq's `gemma2-9b-it`.

---

## ✨ Features

- 🔍 **YOLOv8-based tumor detection** (bounding box annotations)
- 📤 Upload brain scan images (JPG, PNG)
- 📦 Automatic visualization with blue box overlays
- 🤖 LLM-generated **clinical summary report**
- 💾 Downloadable annotated image and diagnostic text
- 🧠 Model: `yolov8-brain-tumor.pt` trained for binary classification (`positive`, `negative`)
- 💬 LLM: Groq `gemma2-9b-it`, structured medical prompt

---

## 🛠 Tech Stack

- `Streamlit` – interactive UI
- `YOLOv8` – real-time tumor detection
- `Pillow` – image drawing/overlay
- `Groq Python SDK` – LLM medical reporting
- `Python` – the glue



