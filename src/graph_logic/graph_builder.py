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

def find_junction_nodes(skeleton: np.ndarray) -> set:
    """
    Identifies junction/endpoint nodes in a skeleton: pixels with
    1 neighbor (endpoints) or 3+ neighbors (intersections). Pixels
    with exactly 2 neighbors are just "pass-through" points on a
    straight/curved road segment, not meaningful graph nodes.
    """
    rows, cols = np.where(skeleton == 1)
    points = set(zip(rows.tolist(), cols.tolist()))

    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),           (0, 1),
                         (1, -1),  (1, 0),  (1, 1)]

    junctions = set()
    for (r, c) in points:
        neighbor_count = sum(
            1 for dr, dc in neighbor_offsets
            if (r + dr, c + dc) in points
        )
        if neighbor_count != 2:
            junctions.add((r, c))

    return junctions


def skeleton_to_junction_graph(skeleton: np.ndarray) -> nx.Graph:
    """
    Builds a compact graph where nodes are ONLY junctions/endpoints
    (not every pixel). Edges connect junctions via the actual path
    length walked along the skeleton between them - this is the
    standard representation used in real road-network analysis
    (matches how OpenStreetMap/Google Maps represent road graphs).

    This keeps the graph small (tens to hundreds of nodes instead of
    thousands), making exact shortest-path computations fast even on
    complex, dense road networks - no approximation/sampling needed.
    """
    pixel_graph = skeleton_to_graph(skeleton)
    junction_nodes = find_junction_nodes(skeleton)

    # keep only junctions that actually exist in the pixel graph
    junction_nodes = junction_nodes & set(pixel_graph.nodes())

    junction_graph = nx.Graph()
    junction_graph.add_nodes_from(junction_nodes)

    visited_edges = set()

    for start_node in junction_nodes:
        for neighbor in pixel_graph.neighbors(start_node):
            if neighbor in junction_nodes:
                # direct junction-to-junction edge (very short segment)
                edge_key = frozenset([start_node, neighbor])
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    junction_graph.add_edge(start_node, neighbor, weight=1.0)
                continue

            # walk along the path until we hit another junction
            path_length = 1.0
            prev_node = start_node
            current_node = neighbor
            visited_in_walk = {start_node}

            while current_node not in junction_nodes:
                visited_in_walk.add(current_node)
                next_candidates = [
                    n for n in pixel_graph.neighbors(current_node)
                    if n != prev_node and n not in visited_in_walk
                ]
                if not next_candidates:
                    break  # dead end mid-walk (shouldn't normally happen)
                prev_node = current_node
                current_node = next_candidates[0]
                path_length += 1.0

            if current_node in junction_nodes and current_node != start_node:
                edge_key = frozenset([start_node, current_node])
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    junction_graph.add_edge(start_node, current_node, weight=path_length)

    return junction_graph

if __name__ == "__main__":
    from src.graph_logic.skeletonize import mask_to_skeleton

    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[45:55, 10:90] = 1

    skeleton = mask_to_skeleton(test_mask)
    graph = skeleton_to_graph(skeleton)

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(graph)}")