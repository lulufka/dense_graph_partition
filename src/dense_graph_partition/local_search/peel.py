from dataclasses import dataclass

import networkx as nx

from dense_graph_partition.core.evaluation import partition_density
from dense_graph_partition.core.types import Node, Partition
from dense_graph_partition.local_search.move import apply_isolate_node, delta_isolate_node
from dense_graph_partition.local_search.result import LocalSearchResult, build_local_search_result
from dense_graph_partition.local_search.state import PartitionState, build_partition_state


@dataclass(frozen=True)
class PeelCandidate:
    """
    Represents a candidate node peel operation.

    Attributes:
        node (Node): Node that should be removed from its current cluster.
        delta (float): Expected change in density. Positive values are improvements.
    """
    node: Node
    delta: float


def delta_peel_node(state: PartitionState, node: Node) -> float:
    """
    Computes the density change when peeling a node into a singleton cluster.

    Args:
        state (PartitionState): Current local-search state.
        node (Node): Node to peel.

    Returns:
        float: Change in density. Positive values are improvements.
    """
    return delta_isolate_node(state, node)


def apply_peel_node(state: PartitionState, node: Node) -> None:
    """
    Peels a node from its current cluster into a new singleton cluster.

    Args:
        state (PartitionState): Current local-search state.
        node (Node): Node to peel.

    Raises:
        ValueError: If the node is already in a singleton cluster.
    """
    apply_isolate_node(state, node)


def apply_peel_candidate(state: PartitionState, candidate: PeelCandidate) -> None:
    """
    Applies a peel candidate to the current state.

    Args:
        state (PartitionState): Current local-search state.
        candidate (PeelCandidate): Peel candidate to apply.
    """
    apply_peel_node(state, candidate.node)


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
        LocalSearchResult: Final partition and local-search statistics.
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

        apply_peel_candidate(state, candidate)
        peel_count += 1

    return build_local_search_result(G, state, initial_score, peel_count, used_passes)
