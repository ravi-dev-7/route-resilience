"""
graph_builder.py
Converts a road skeleton (binary image) into a NetworkX graph.
Each skeleton pixel becomes a node; adjacent skeleton pixels are connected by edges.
"""

import networkx as nx
import numpy as np


def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """
    Build a graph from a skeletonized binary image.

    Args:
        skeleton: 2D numpy array, skeleton pixels = 1, rest = 0

    Returns:
        networkx.Graph where nodes are (row, col) pixel coordinates
        and edges connect 8-neighbor adjacent skeleton pixels,
        weighted by Euclidean distance.
    """
    G = nx.Graph()
    rows, cols = np.where(skeleton == 1)
    points = set(zip(rows.tolist(), cols.tolist()))

    for (r, c) in points:
        G.add_node((r, c))

    # 8-connected neighborhood offsets
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),           (0, 1),
                         (1, -1),  (1, 0),  (1, 1)]

    for (r, c) in points:
        for dr, dc in neighbor_offsets:
            neighbor = (r + dr, c + dc)
            if neighbor in points:
                dist = (dr ** 2 + dc ** 2) ** 0.5
                G.add_edge((r, c), neighbor, weight=dist)

    return G


if __name__ == "__main__":
    from src.graph_logic.skeletonize import mask_to_skeleton

    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[45:55, 10:90] = 1

    skeleton = mask_to_skeleton(test_mask)
    graph = skeleton_to_graph(skeleton)

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(graph)}")