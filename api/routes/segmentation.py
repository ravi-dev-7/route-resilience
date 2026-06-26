"""
segmentation.py
Endpoint for running the trained U-Net+ResNet34 model on an uploaded
satellite image to produce a road segmentation mask.
"""

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
import base64

from src.models.unet_resnet import UNetResNet34
import albumentations as A
from albumentations.pytorch import ToTensorV2

router = APIRouter()

# load model once at startup (lazy-loaded on first request)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = UNetResNet34(pretrained=False)
        # NOTE: load trained weights once training is done:
        _model.load_state_dict(torch.load("src/models/checkpoints/best_model.pth", map_location="cpu"))
        _model.eval()
    return _model


def preprocess_image(image: np.ndarray, size: int = 512):
    transform = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    augmented = transform(image=image)
    return augmented["image"].unsqueeze(0)  # add batch dim


@router.post("/predict")
async def predict_road_mask(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = get_model()
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        prob_mask = torch.sigmoid(output).squeeze().numpy()
        binary_mask = (prob_mask > 0.5).astype(np.uint8) * 255

    _, encoded_mask = cv2.imencode(".png", binary_mask)
    mask_base64 = base64.b64encode(encoded_mask).decode("utf-8")

    return JSONResponse({
        "status": "success",
        "mask_base64": mask_base64,
        "road_pixel_count": int((binary_mask > 0).sum()),
    })