"""
graph.py
Endpoint that takes a predicted road mask and runs the full graph
pipeline: skeletonize -> build graph -> heal gaps -> return stats.
"""

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
import base64
import networkx as nx

from src.models.unet_resnet import UNetResNet34
from src.graph_logic.skeletonize import mask_to_skeleton
from src.graph_logic.graph_builder import skeleton_to_junction_graph
from src.graph_logic.healing import heal_graph
import albumentations as A
from albumentations.pytorch import ToTensorV2

router = APIRouter()

_model = None


def get_model():
    global _model
    if _model is None:
        _model = UNetResNet34(pretrained=False)
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
    return augmented["image"].unsqueeze(0)


@router.post("/graph-analysis")
async def analyze_graph(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = get_model()
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        prob_mask = torch.sigmoid(output).squeeze().numpy()
        binary_mask = (prob_mask > 0.5).astype(np.uint8)

    skeleton = mask_to_skeleton(binary_mask)
    raw_graph = skeleton_to_junction_graph(skeleton)
    components_before = nx.number_connected_components(raw_graph)

    healed_graph = heal_graph(raw_graph, max_bridge_distance=25.0)
    components_after = nx.number_connected_components(healed_graph)

    skeleton_vis = (skeleton * 255).astype(np.uint8)
    _, encoded_skeleton = cv2.imencode(".png", skeleton_vis)
    skeleton_base64 = base64.b64encode(encoded_skeleton).decode("utf-8")

    return JSONResponse({
        "status": "success",
        "skeleton_base64": skeleton_base64,
        "total_nodes": healed_graph.number_of_nodes(),
        "total_edges": healed_graph.number_of_edges(),
        "components_before_healing": components_before,
        "components_after_healing": components_after,
        "bridges_added": components_before - components_after,
    })