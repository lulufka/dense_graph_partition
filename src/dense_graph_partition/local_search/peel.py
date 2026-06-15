from dataclasses import dataclass

import networkx as nx

from dense_graph_partition.core.evaluation import partition_density
from dense_graph_partition.core.types import Node, Partition
from dense_graph_partition.local_search.result import LocalSearchResult
from dense_graph_partition.local_search.search import build_local_search_result
from dense_graph_partition.local_search.state import PartitionState, neighbors_in_cluster, build_partition_state


@dataclass(frozen=True)
class PeelCandidate:
    """
    Represents a candidate node peel operation.

    Attributes:
        node (Node): Node that should be removed from its current cluster.
        delta (float): Expected change in density.
    """
    node: Node
    delta: float


def delta_peel_node(state: PartitionState, node: Node) -> float:
    """
    Computes the density change when peeling a node into a singleton cluster.

    Args:
        state (PartitionState): Current local-search state.
        node (Node): Node to peel

    Returns:
        float: Change in density. Positive values are improvements.
    """
    source_cluster = state.cluster_of[node]
    source_size = state.cluster_sizes[source_cluster]

    if source_size <= 1:
        return float("-inf")

    source_edges = state.internal_edges[source_cluster]
    degree_inside_source = neighbors_in_cluster(state, node, source_cluster)

    old_score = source_edges / source_size
    new_score = (source_edges - degree_inside_source) / (source_size - 1)

    return new_score - old_score


def apply_peel_node(state: PartitionState, node: Node) -> None:
    """
    Peels a node from its current cluster into a new singleton cluster.

    Args:
        state (PartitionState): Current local-search state.
        node (Node): Node to peel

    Raises:
        ValueError: If the node is already in a singleton cluster.
    """
    source_cluster = state.cluster_of[node]

    if state.cluster_sizes[source_cluster] <= 1:
        raise ValueError("Cannot peel a node that already is a singleton.")

    degree_inside_source = neighbors_in_cluster(state, node, source_cluster)

    state.internal_edges[source_cluster] -= degree_inside_source
    state.cluster_sizes[source_cluster] -= 1
    state.clusters[source_cluster].remove(node)

    new_cluster_index = len(state.clusters)
    state.clusters.append({node})
    state.cluster_sizes.append(1)
    state.internal_edges.append(0)
    state.cluster_of[node] = new_cluster_index


def best_peel_node(state: PartitionState, epsilon: float = 1e-12) -> PeelCandidate | None:
    """
    Finds the best improving peel operation over all nodes.

    Args:
        state (PartitionState): Current local-search state.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        PeelCandidate | None: Best improving peel candidate, or ``None`` if no improving peel exists.
    """
    best: PeelCandidate | None = None

    for node in state.G.nodes():
        delta = delta_peel_node(state, node)

        if delta <= epsilon:
            continue

        candidate = PeelCandidate(node, delta)

        if best is None or candidate.delta > best.delta:
            best = candidate

    return best


def refine_partition_peel_node(G: nx.Graph, partition: Partition, max_passes: int = 1000, epsilon: float = 1e-12) -> LocalSearchResult:
    """
    Refines a partition using repeated best-improvement node peeling.

    Args:
        G (nx.Graph): Input graph.
        partition (Partition): Initial partition.
        max_passes (int): Maximum number of peel iterations.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        LocalSearchResult: Final partition and local-search statistic.
    """
    state = build_partition_state(G, partition)
    initial_score = partition_density(G, partition)

    peel_count = 0
    used_passes = 0

    for _ in range(max_passes):
        used_passes += 1

        candidate = best_peel_node(state, epsilon)

        if candidate is None:
            break

        apply_peel_node(state, candidate.node)
        peel_count += 1

    return build_local_search_result(G, state, initial_score, peel_count, used_passes)