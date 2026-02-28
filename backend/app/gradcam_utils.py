import torch
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .inference import model, transform

device = "cpu"

# ---------------------------------
# Select target layer for GradCAM
# (last conv layer in CNN head)
# ---------------------------------
target_layers = [model.cnn.conv[3]]


def generate_gradcam(image: Image.Image):
    """
    Generates GradCAM heatmap and returns image (numpy array)
    """

    # preprocess image
    input_tensor = transform(image).unsqueeze(0)

    # forward pass to get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_class = torch.argmax(outputs, dim=1).item()

    # GradCAM setup
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # convert original image to numpy
    image_np = np.array(image.resize((224,224))) / 255.0

    # overlay heatmap
    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    # convert back to PIL image
    result_image = Image.fromarray(visualization)

    return result_image