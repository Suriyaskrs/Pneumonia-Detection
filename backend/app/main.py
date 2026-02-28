from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import io

from .inference import predict_image
from .gradcam_utils import generate_gradcam
from .schemas import PredictionResponse, HealthResponse

# ---------------------------------
# Create FastAPI app
# ---------------------------------
app = FastAPI(
    title="Pneumonia Detection API",
    description="Hybrid ViT-CNN model with Explainable AI",
    version="1.0"
)

# ---------------------------------
# Health Check
# ---------------------------------
@app.get("/", response_model=HealthResponse)
def health_check():
    return {"message": "API is running successfully 🚀"}

# ---------------------------------
# Prediction Endpoint
# ---------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    label, confidence = predict_image(image)

    return {
        "prediction": label,
        "confidence": confidence
    }

# ---------------------------------
# GradCAM Endpoint
# ---------------------------------
@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    """
    Returns GradCAM heatmap image
    """

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    heatmap_image = generate_gradcam(image)

    img_bytes = io.BytesIO()
    heatmap_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")