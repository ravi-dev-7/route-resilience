"""
centrality.py
Computes Betweenness Centrality on the road graph to identify
"Gatekeeper Nodes" - critical junctions where many shortest paths pass through.
High centrality = removing this node disrupts a lot of routing.
"""

import networkx as nx


def compute_betweenness_centrality(G: nx.Graph) -> dict:
    """
    Compute betweenness centrality for every node in the graph.

    Args:
        G: input road graph (should be connected, or run on largest component)

    Returns:
        dict mapping node -> centrality score (0 to 1)
    """
    centrality = nx.betweenness_centrality(G, weight="weight", normalized=True)
    return centrality


def get_gatekeeper_nodes(G: nx.Graph, top_n: int = 5) -> list:
    """
    Returns the top_n nodes with highest betweenness centrality -
    these are the "Gatekeeper Nodes" most critical to network connectivity.

    Args:
        G: input road graph
        top_n: how many top nodes to return

    Returns:
        list of (node, centrality_score) tuples, sorted descending by score
    """
    centrality = compute_betweenness_centrality(G)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_n]


if __name__ == "__main__":
    from src.graph_logic.skeletonize import mask_to_skeleton
    from src.graph_logic.graph_builder import skeleton_to_graph
    import numpy as np

    # build a simple cross-shaped road network (one obvious junction)
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[45:55, 10:90] = 1   # horizontal road
    test_mask[10:90, 45:55] = 1   # vertical road (creates a junction in the middle)

    skeleton = mask_to_skeleton(test_mask)
    graph = skeleton_to_graph(skeleton)

    print(f"Total nodes: {graph.number_of_nodes()}")

    gatekeepers = get_gatekeeper_nodes(graph, top_n=5)

    print("Top 5 Gatekeeper Nodes (node, centrality_score):")
    for node, score in gatekeepers:
        print(f"  {node} -> {score:.4f}")