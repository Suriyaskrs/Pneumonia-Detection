import streamlit as st
import requests
from PIL import Image
from fpdf import FPDF
import io

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Pneumonia Detection", layout="wide")

# ---------------- UI HEADER ----------------
st.markdown("""
# 🫁 Pneumonia Detection Dashboard
Upload chest X-rays and get predictions with GradCAM explanations.
""")

st.divider()

# ---------------- MULTIPLE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload Chest X-rays",
    type=["jpg","png","jpeg"],
    accept_multiple_files=True
)

results_for_pdf = []

# ---------------- FUNCTION: CREATE PDF ----------------
def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for res in results:
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Pneumonia Detection Report", ln=True)

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Prediction: {res['prediction']}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {res['confidence']:.2f}", ln=True)

        img_path = res["image_path"]
        pdf.image(img_path, x=10, y=40, w=180)

    return pdf.output(dest="S").encode("latin-1")

# ---------------- PREDICT BUTTON ----------------
if uploaded_files:
    if st.button("🔍 Run Prediction"):
        for idx, uploaded_file in enumerate(uploaded_files):
            st.divider()

            col1, col2 = st.columns([1,1])

            image = Image.open(uploaded_file).convert("RGB")
            img_bytes = uploaded_file.getvalue()

            # ---- Prediction ----
            files = {"file": img_bytes}
            response = requests.post(f"{API_URL}/predict", files=files)
            result = response.json()

            # ---- GradCAM ----
            heatmap_response = requests.post(f"{API_URL}/gradcam", files=files)

            # ---- SHOW ORIGINAL IMAGE ----
            with col1:
                st.image(image, caption=f"Image {idx+1}", use_column_width=True)

            # ---- SHOW RESULTS ----
            with col2:
                st.markdown(f"""
                ### 🧠 Prediction
                **Label:** `{result['prediction']}`  
                **Confidence:** `{result['confidence']:.2f}`
                """)

                st.image(heatmap_response.content, caption="GradCAM Heatmap")

            # Save for PDF
            temp_path = f"temp_{idx}.png"
            image.save(temp_path)

            results_for_pdf.append({
                "prediction": result['prediction'],
                "confidence": result['confidence'],
                "image_path": temp_path
            })

# ---------------- PDF DOWNLOAD ----------------
if results_for_pdf:
    st.divider()
    pdf_bytes = create_pdf(results_for_pdf)

    st.download_button(
        label="📄 Download Report as PDF",
        data=pdf_bytes,
        file_name="pneumonia_report.pdf",
        mime="application/pdf"
    )