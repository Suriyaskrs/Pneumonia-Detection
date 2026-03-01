# 🫁 Explainable Pneumonia Detection using Hybrid ViT + CNN

A production-ready deep learning web application that detects pneumonia from chest X-ray images using a **hybrid Vision Transformer + CNN architecture** with **Explainable AI (Grad-CAM & SHAP)** and a **full-stack deployment (FastAPI + Streamlit)**.

---

## 🌍 Project Overview

This project provides an end-to-end pipeline for **medical image classification** with explainability and deployment.

### Key capabilities

* Pneumonia detection from chest X-ray images
* Explainable AI heatmaps (Region of Interest)
* Multi-image upload support
* Automated PDF medical report generation
* REST API backend + Web UI frontend
* Ready for cloud deployment

---

## 🧠 Motivation

Medical imaging models are often **black boxes**.
This project focuses on **Explainable AI (XAI)** so doctors can understand:

* **Why** the model predicted pneumonia
* **Where** the model looked in the lungs
* **Which features influenced the decision**

Goal → bridge the gap between **AI accuracy** and **clinical trust**.

---

## 🏗️ System Architecture

```
Chest X-ray Image
        ↓
Vision Transformer (Feature Extractor)
        ↓
768-Dim Embeddings
        ↓
CNN Classifier Head
        ↓
Prediction (Normal / Pneumonia)
        ↓
Grad-CAM Heatmap + PDF Report
```

---

## 🤖 Model Architecture

Hybrid deep learning model:

| Component             | Role                             |
| --------------------- | -------------------------------- |
| **ViT-Base (Frozen)** | Extract global image features    |
| **Linear Layer**      | Convert embeddings → feature map |
| **CNN Head**          | Learn spatial pneumonia patterns |
| **Grad-CAM**          | Visual explanation (ROI)         |
| **SHAP**              | Feature importance analysis      |

---

## 📊 Datasets Used

Model trained and evaluated on multiple datasets:

1. **RSNA Pneumonia Detection Challenge**
2. **Chest X-Ray Pneumonia Dataset (Kaggle)**
3. **PneumoniaMNIST**

Multi-dataset evaluation improves generalization and research value.

---

## 📈 Model Performance

| Model                | Accuracy  |
| -------------------- | --------- |
| ResNet50 Baseline    | 86%       |
| Vision Transformer   | 86%       |
| **Hybrid ViT + CNN** | **87% ⭐** |

---

## ✨ Features

* Multi-image inference
* Grad-CAM heatmap visualization
* SHAP feature importance analysis
* Downloadable medical PDF report
* REST API for integration
* Web UI for easy usage
* Cloud deployment ready

---

## 🗂️ Project Structure

```
Pneumonia-Detection/
│
├── backend/
│   ├── app/                # FastAPI backend
│   ├── model_weights/      # Trained model (Git LFS)
│   ├── requirements.txt
│   └── runtime.txt
│
├── frontend/
│   ├── app.py              # Streamlit UI
│   └── requirements.txt
│
└── README.md
```

---

# ⚙️ Backend Setup (FastAPI)

## 1️⃣ Clone repository

```
git clone https://github.com/yourusername/Pneumonia-Detection.git
cd Pneumonia-Detection/backend
```

## 2️⃣ Create environment (Conda recommended)

```
conda create -n pneumonia-api python=3.10 -y
conda activate pneumonia-api
```

## 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

## 4️⃣ Run backend server

```
uvicorn app.main:app --reload
```

API available at:

```
http://127.0.0.1:8000/docs
```

Swagger UI will open automatically.

---

# 🎨 Frontend Setup (Streamlit)

## 1️⃣ Navigate to frontend folder

```
cd ../frontend
```

## 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

## 3️⃣ Run Streamlit app

```
streamlit run app.py
```

Web app available at:

```
http://localhost:8501
```

---

## 📄 PDF Report Output

Each uploaded image generates a report containing:

* Prediction & confidence score
* Original X-ray image
* Grad-CAM heatmap

Designed to mimic a **clinical report workflow**.

---

## 🔬 Explainable AI

### 🔥 Grad-CAM

Highlights lung regions influencing prediction.

### 📊 SHAP

Identifies important ViT embedding features.

Together they provide **model transparency**.

---

## 🌐 Deployment

| Component        | Platform         |
| ---------------- | ---------------- |
| Backend API      | Render / Railway |
| Frontend Web App | Streamlit Cloud  |

---

## 👨‍💻 Tech Stack

* PyTorch
* Vision Transformers (timm)
* FastAPI
* Streamlit
* Grad-CAM
* SHAP
* Git LFS

---

## 🚀 Future Improvements

* DICOM support
* Mobile app integration
* Model compression for edge devices
* Clinical validation studies

---

## 📜 License

MIT License
