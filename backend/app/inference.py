import torch
from PIL import Image
import torchvision.transforms as transforms
from .model import load_model

# -----------------------------
# DEVICE
# -----------------------------
device = "cpu"

# -----------------------------
# LOAD MODEL ONCE (startup)
# -----------------------------
MODEL_PATH = "model_weights/hybrid_rsna.pth"
model = load_model(MODEL_PATH, device)

# -----------------------------
# IMAGE PREPROCESSING
# Must match training transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(image: Image.Image):
    """
    Takes PIL image → returns prediction + confidence
    """

    # preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

    label = "PNEUMONIA" if pred_class == 1 else "NORMAL"

    return label, float(confidence)