"""
resilience.py
Simulates node ablation (removing a "gatekeeper" node) and measures
the resulting impact using ISRO's exact Resilience Index definition:

Resilience Index (R) = avg_shortest_path_length(baseline) / avg_shortest_path_length(perturbed)

A lower R indicates a highly vulnerable network (perturbed network's
average shortest path length explodes compared to baseline).
"""

import networkx as nx
import copy


def average_shortest_path_length_safe(G: nx.Graph) -> float:
    """
    Computes average shortest path length across the LARGEST connected
    component (since a disconnected graph has undefined path length
    between components). This is standard practice for resilience analysis.
    """
    if G.number_of_nodes() == 0:
        return float("inf")

    largest_cc_nodes = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc_nodes)

    if subgraph.number_of_nodes() <= 1:
        return 0.0

    return nx.average_shortest_path_length(subgraph, weight="weight")

def simulate_node_ablation(G: nx.Graph, node_to_remove) -> dict:
    """
    Remove a node from the graph and measure the change in average
    shortest path length (baseline vs perturbed).

    If the removal fragments the network, the average shortest path
    length is computed only on pairs that are still reachable, but
    the loss of connectivity itself is factored into the Resilience
    Index so fragmentation is correctly penalized (not hidden).
    """
    baseline_aspl = average_shortest_path_length_safe(G)
    before_components = nx.number_connected_components(G)
    before_largest = len(max(nx.connected_components(G), key=len))
    total_nodes_before = G.number_of_nodes()

    G_ablated = copy.deepcopy(G)
    G_ablated.remove_node(node_to_remove)

    if G_ablated.number_of_nodes() == 0:
        perturbed_aspl = float("inf")
        after_components = 0
        after_largest = 0
    else:
        perturbed_aspl = average_shortest_path_length_safe(G_ablated)
        after_components = nx.number_connected_components(G_ablated)
        after_largest = len(max(nx.connected_components(G_ablated), key=len))

    # Reachability penalty: fraction of nodes that fell OUT of the
    # largest reachable component after ablation. 0 = no nodes lost,
    # 1 = network completely fragmented into isolated pieces.
    reachability_loss = 1 - (after_largest / total_nodes_before) if total_nodes_before > 0 else 1.0

    # ISRO's formula: R = baseline / perturbed, but we apply the
    # reachability penalty multiplicatively so fragmentation cannot
    # be masked by a shorter path length inside a shrunk component.
    if perturbed_aspl == 0 or perturbed_aspl == float("inf"):
        raw_ratio = 0.0
    else:
        raw_ratio = baseline_aspl / perturbed_aspl

    resilience_index = round(raw_ratio * (1 - reachability_loss), 4)

    return {
        "node_removed": node_to_remove,
        "baseline_avg_shortest_path": round(baseline_aspl, 4),
        "perturbed_avg_shortest_path": round(perturbed_aspl, 4) if perturbed_aspl != float("inf") else "disconnected",
        "components_before": before_components,
        "components_after": after_components,
        "largest_component_before": before_largest,
        "largest_component_after": after_largest,
        "reachability_loss_pct": round(reachability_loss * 100, 2),
        "resilience_index": resilience_index,
    }



def compute_resilience_index(G: nx.Graph, gatekeeper_nodes: list) -> dict:
    """
    Runs ablation on each gatekeeper node using ISRO's exact Resilience
    Index formula. Returns per-node results and the overall (average)
    resilience index across all tested gatekeeper nodes.

    R = 1.0  -> no impact from removing this node (very robust)
    R < 1.0  -> perturbed path length grew vs baseline (vulnerable)
    R -> 0   -> network fragmented/disconnected (highly vulnerable)
    """
    results = []
    total_resilience = 0.0

    for node, score in gatekeeper_nodes:
        result = simulate_node_ablation(G, node)
        result["centrality_score"] = round(score, 4)
        results.append(result)
        total_resilience += result["resilience_index"]

    avg_resilience_index = round(total_resilience / len(gatekeeper_nodes), 4) if gatekeeper_nodes else 0.0

    return {
        "per_node_results": results,
        "overall_resilience_index": avg_resilience_index,
    }


if __name__ == "__main__":
    from src.graph_logic.skeletonize import mask_to_skeleton
    from src.graph_logic.graph_builder import skeleton_to_graph
    from src.graph_logic.centrality import get_gatekeeper_nodes
    import numpy as np

    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[45:55, 10:90] = 1
    test_mask[10:90, 45:55] = 1

    skeleton = mask_to_skeleton(test_mask)
    graph = skeleton_to_graph(skeleton)

    gatekeepers = get_gatekeeper_nodes(graph, top_n=3)

    report = compute_resilience_index(graph, gatekeepers)

    print("Resilience Report (ISRO formula: baseline_ASPL / perturbed_ASPL):")
    for r in report["per_node_results"]:
        print(f"  Node {r['node_removed']} | Centrality: {r['centrality_score']} | "
              f"Baseline ASPL: {r['baseline_avg_shortest_path']} | "
              f"Perturbed ASPL: {r['perturbed_avg_shortest_path']} | "
              f"R: {r['resilience_index']} | "
              f"Components: {r['components_before']} -> {r['components_after']}")

    print(f"\nOverall Resilience Index: {report['overall_resilience_index']}")