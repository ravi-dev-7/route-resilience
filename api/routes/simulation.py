"""
simulation.py
Endpoint that runs Betweenness Centrality + Node Ablation on the
healed road graph, returning the Resilience Index using ISRO's
exact formula (baseline ASPL / perturbed ASPL, fragmentation-penalized).
"""

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
import networkx as nx

from src.models.unet_resnet import UNetResNet34
from src.graph_logic.skeletonize import mask_to_skeleton
from src.graph_logic.graph_builder import skeleton_to_junction_graph
from src.graph_logic.healing import heal_graph
from src.graph_logic.centrality import get_gatekeeper_nodes
from src.graph_logic.resilience import compute_resilience_index
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


@router.post("/simulate-resilience")
async def simulate_resilience(file: UploadFile = File(...), top_n: int = 5):
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
    healed_graph = heal_graph(raw_graph, max_bridge_distance=25.0)

    # use only the largest connected component for centrality/ablation
    largest_cc_nodes = max(nx.connected_components(healed_graph), key=len)
    main_graph = healed_graph.subgraph(largest_cc_nodes).copy()

    if main_graph.number_of_nodes() < 3:
        return JSONResponse({
            "status": "error",
            "message": "Graph too small for resilience analysis (need at least 3 connected nodes).",
        }, status_code=400)

    gatekeepers = get_gatekeeper_nodes(main_graph, top_n=top_n)
    report = compute_resilience_index(main_graph, gatekeepers)

    per_node_results = [
        {
            "node": str(r["node_removed"]),
            "centrality_score": r["centrality_score"],
            "resilience_index": r["resilience_index"],
            "components_before": r["components_before"],
            "components_after": r["components_after"],
            "reachability_loss_pct": r["reachability_loss_pct"],
        }
        for r in report["per_node_results"]
    ]

    return JSONResponse({
        "status": "success",
        "total_nodes_analyzed": main_graph.number_of_nodes(),
        "gatekeeper_nodes": per_node_results,
        "overall_resilience_index": report["overall_resilience_index"],
    })