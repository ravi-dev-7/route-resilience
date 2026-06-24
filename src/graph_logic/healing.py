"""
healing.py
Heals broken/disconnected road graph components using a true
Minimum Spanning Tree (MST) + Disjoint Set (Union-Find) approach,
as specified by ISRO:

1. Build a "candidate bridge graph" between disconnected components,
   weighted by Euclidean distance between closest node pairs.
2. Use Disjoint Set to track which components are already merged.
3. Run Kruskal-style MST selection over candidate bridges - only add
   the minimum-weight bridge needed to connect each pair of components.
4. Use angular alignment to reject "unnatural" bridges (e.g. a gap
   that would force a sharp 90-degree turn is likely not a real road).
"""

import networkx as nx
import numpy as np
from itertools import combinations


class DisjointSet:
    """Union-Find structure to track which components are merged."""

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item_a, item_b):
        root_a, root_b = self.find(item_a), self.find(item_b)
        if root_a != root_b:
            self.parent[root_a] = root_b
            return True  # merge happened
        return False  # already in same set


def get_local_direction(G: nx.Graph, node, component_nodes: set, lookback: int = 5):
    """
    Estimate the direction a road segment is heading near 'node' by
    looking at a few nodes back along the component (via BFS), so we
    can check angular alignment at the gap.
    """
    visited = {node}
    queue = [node]
    path = [node]

    while queue and len(path) < lookback:
        current = queue.pop(0)
        for neighbor in G.neighbors(current):
            if neighbor in component_nodes and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                path.append(neighbor)
                break  # follow a single chain, not branching

    if len(path) < 2:
        return None  # not enough points to estimate direction

    start, end = path[0], path[-1]
    direction = np.array([end[0] - start[0], end[1] - start[1]])
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else None


def angular_alignment_score(dir_a, dir_b, bridge_vector):
    """
    Returns a score in [0, 1]: how well the bridge continues the
    natural trajectory of both road segments. 1 = perfectly straight
    continuation, 0 = sharp/unnatural turn.
    """
    if dir_a is None or dir_b is None:
        return 0.5  # neutral score if direction can't be estimated

    bridge_norm = bridge_vector / np.linalg.norm(bridge_vector) if np.linalg.norm(bridge_vector) > 0 else bridge_vector
    cos_a = abs(np.dot(dir_a, bridge_norm))
    cos_b = abs(np.dot(dir_b, bridge_norm))
    return (cos_a + cos_b) / 2


def heal_graph(G: nx.Graph, max_bridge_distance: float = 25.0, min_angular_score: float = 0.3) -> nx.Graph:
    """
    Heals a disconnected road graph using MST + Disjoint Set, with
    Euclidean distance and angular alignment as the bridge cost criteria.

    Args:
        G: input graph (possibly disconnected)
        max_bridge_distance: max pixel distance to consider a candidate bridge
        min_angular_score: minimum alignment score (0-1) to accept a bridge;
                            bridges below this are rejected as "unnatural"

    Returns:
        Healed graph with MST-selected bridging edges added.
    """
    healed = G.copy()
    components = [set(c) for c in nx.connected_components(healed)]

    if len(components) <= 1:
        print("Bridges added: 0 (graph already connected)")
        return healed

    # Disjoint Set over component indices
    comp_ids = list(range(len(components)))
    dsu = DisjointSet(comp_ids)

    # Build all candidate bridges (closest node pair between every
    # component pair), each weighted by distance and penalized by
    # poor angular alignment.
    candidate_bridges = []  # (cost, comp_i, comp_j, node_a, node_b)

    for i, j in combinations(range(len(components)), 2):
        comp_a, comp_b = components[i], components[j]

        min_dist = float("inf")
        best_pair = None

        for node_a in comp_a:
            for node_b in comp_b:
                dist = ((node_a[0] - node_b[0]) ** 2 + (node_a[1] - node_b[1]) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (node_a, node_b)

        if min_dist <= max_bridge_distance and best_pair:
            node_a, node_b = best_pair
            dir_a = get_local_direction(healed, node_a, comp_a)
            dir_b = get_local_direction(healed, node_b, comp_b)
            bridge_vector = np.array([node_b[0] - node_a[0], node_b[1] - node_a[1]])

            alignment = angular_alignment_score(dir_a, dir_b, bridge_vector)

            if alignment >= min_angular_score:
                # cost = distance penalized by poor alignment (MST edge weight)
                cost = min_dist / max(alignment, 0.1)
                candidate_bridges.append((cost, i, j, node_a, node_b, min_dist, alignment))

    # Kruskal-style MST: sort candidates by cost, add only if it
    # merges two previously-separate components (Disjoint Set check)
    candidate_bridges.sort(key=lambda x: x[0])

    bridges_added = 0
    for cost, i, j, node_a, node_b, dist, alignment in candidate_bridges:
        if dsu.find(i) != dsu.find(j):
            healed.add_edge(node_a, node_b, weight=dist, healed=True, angular_score=round(alignment, 3))
            dsu.union(i, j)
            bridges_added += 1

    print(f"Bridges added: {bridges_added} (MST + Disjoint Set, angular-alignment filtered)")
    return healed


if __name__ == "__main__":
    from src.graph_logic.skeletonize import mask_to_skeleton
    from src.graph_logic.graph_builder import skeleton_to_graph

    # simulate a road with a gap (occlusion) in the middle - straight road
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[45:55, 10:40] = 1   # left segment
    test_mask[45:55, 55:90] = 1   # right segment (gap from 40 to 55)

    skeleton = mask_to_skeleton(test_mask)
    graph = skeleton_to_graph(skeleton)

    print(f"Before healing - Connected components: {nx.number_connected_components(graph)}")

    healed_graph = heal_graph(graph, max_bridge_distance=25.0)

    print(f"After healing - Connected components: {nx.number_connected_components(healed_graph)}")