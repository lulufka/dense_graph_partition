import random
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from dense_graph_partition.core.evaluation import partition_density
from dense_graph_partition.core.types import Node, Partition
from dense_graph_partition.local_search.result import LocalSearchResult
from dense_graph_partition.local_search.state import PartitionState, neighbors_in_cluster, state_to_partition, \
    build_partition_state


@dataclass(frozen=True)
class MoveCandidate:
    """
    Represents a candidate node move.

    Attributes:
        node (Node): Node that should be moved.
        target_cluster (int | None): Destination cluster. If ``None``, the moves isolates the node as a singleton cluster.
        delta (float): Expected change in density.
        isolate (bool): Whether the node should be isolated instead of moved to an existing cluster.
    """
    node: Node
    target_cluster: int | None
    delta: float
    isolate: bool = False


def delta_isolate_node(state: PartitionState, node: Node) -> float:
    """
    Computes the score change when moving a node into a singleton cluster.

    Args:
        state (PartitionState): Current local-search state.
        node (Node): Node to isolate.

    Returns:
        float: Change in density score.
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


def apply_isolate_node(state: PartitionState, node: Node) -> None:
    """
    Moves a node into a new singleton cluster.

    Args:
        state (PartitionState): Current local-search state.
        node (Node): Node to isolate.

    Raises:
        ValueError: If the node is already a singleton.
    """
    source_cluster = state.cluster_of[node]

    if state.cluster_sizes[source_cluster] <= 1:
        raise ValueError("Cannot isolate a node that already is a singleton")

    degree_inside_source = neighbors_in_cluster(state, node, source_cluster)

    state.internal_edges[source_cluster] -= degree_inside_source
    state.cluster_sizes[source_cluster] -= 1
    state.clusters[source_cluster].remove(node)

    state.clusters.append({node})
    state.cluster_sizes.append(1)
    state.internal_edges.append(0)
    state.cluster_of[node] = len(state.clusters) - 1


def delta_move_node(state: PartitionState, node: Node, target_cluster: int) -> float:
    """
    Computes the density score change when moving a node to another cluster.

    Args:
        state (PartitionState): Current local-search state.
        node (Node): Node to move.
        target_cluster (int): Destination cluster index.

    Returns:
        float: The change in density. Positive values are improvements.
    """
    source_cluster = state.cluster_of[node]

    if source_cluster == target_cluster:
        return float("-inf")

    source_size = state.cluster_sizes[source_cluster]
    target_size = state.cluster_sizes[target_cluster]

    if target_size == 0:
        return float("-inf")

    source_edges = state.internal_edges[source_cluster]
    target_edges = state.internal_edges[target_cluster]

    degree_inside_source = neighbors_in_cluster(state, node, source_cluster)
    degree_inside_target = neighbors_in_cluster(state, node, target_cluster)

    old_source_score = source_edges / source_size
    old_target_score = target_edges / target_size

    new_target_score = (target_edges + degree_inside_target) / (target_size + 1)

    if source_size == 1:
        new_source_score = 0.0
    else:
        new_source_score = (source_edges - degree_inside_source) / (source_size - 1)

    return (new_source_score + new_target_score) - (old_source_score + old_target_score)



def apply_move_node(state: PartitionState, node: Node, target_cluster: int) -> None:
    """
    Executes a node move and updates the partition state in place.

    Args:
        state (PartitionState): Current local-search state.
        node (Node): Node to move.
        target_cluster (int): Destination cluster index.

    Raises:
        ValueError: If source and target cluster are identical or the target cluster is empty.
    """
    source_cluster = state.cluster_of[node]

    if source_cluster == target_cluster:
        raise ValueError("Source and target cluster are identical.")

    if state.cluster_sizes[target_cluster] == 0:
        raise ValueError("Cannot move node into an empty cluster.")

    degree_inside_source = neighbors_in_cluster(state, node, source_cluster)
    degree_inside_target = neighbors_in_cluster(state, node, target_cluster)

    state.internal_edges[source_cluster] -= degree_inside_source
    state.internal_edges[target_cluster] += degree_inside_target

    state.cluster_sizes[source_cluster] -= 1
    state.cluster_sizes[target_cluster] += 1

    state.clusters[source_cluster].remove(node)
    state.clusters[target_cluster].add(node)

    state.cluster_of[node] = target_cluster


def best_move_for_node(state: PartitionState, node: Node) -> MoveCandidate | None:
    """
    Finds the best move candidate for a node.
    The neighborhood consists of moves to neighboring clusters and the option to isolate the node as a singleton cluster.

    Args:
        state (PartitionState): Current local-search state.
        node (Node): Node to evaluate.

    Returns:
        MoveCandidate | None: Best candidate move, or ``None`` if no valid move exists.
    """
    source_cluster = state.cluster_of[node]

    candidate_clusters = {
        state.cluster_of[neighbor] for neighbor in state.G.neighbors(node) if state.cluster_of[neighbor] != source_cluster
    }

    best: MoveCandidate | None = None

    for target_cluster in candidate_clusters:
        delta = delta_move_node(state, node, target_cluster)

        if best is None or delta > best.delta:
            best = MoveCandidate(node, target_cluster, delta, isolate=False)

    isolate_delta = delta_isolate_node(state, node)
    if best is None or isolate_delta > best.delta:
        best = MoveCandidate(node, None, isolate_delta, isolate=True)

    if best.delta == float("-inf"):
        return None

    return best


def apply_move_candidate(state: PartitionState, candidate: MoveCandidate) -> None:
    """
    Applies a move candidate to the current state.

    Args:
        state (PartitionState): Current local-search state.
        candidate (MoveCandidate): Move candidate to apply.

    Raises:
        ValueError: If the candidate has no valid target cluster.
    """
    if candidate.isolate:
        apply_isolate_node(state, candidate.node)
        return

    if candidate.target_cluster is None:
        raise ValueError("Move candidate has no target cluster.")

    apply_move_node(state, candidate.node, candidate.target_cluster)


def _result(G: nx.Graph, state: PartitionState, initial_score: float, num_moves: int, num_passes: int) -> LocalSearchResult:
    final_partition = state_to_partition(state)
    final_score = partition_density(G, final_partition)
    return LocalSearchResult(final_partition, num_moves, num_passes, initial_score, final_score)


def refine_partition_move_first(
        G: nx.Graph,
        partition: Partition,
        max_passes: int = 1000,
        random_seed: Optional[int] = None,
        epsilon: float = 1e-12
) -> LocalSearchResult:
    """
    Refines a partition using first-improvement node moves.

    Args:
        G (nx.Graph): Input graph.
        partition (Partition): Initial partition.
        max_passes (int): Maximum number of passes over all nodes.
        random_seed (Optional[int]): Random seed for node order shuffling.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        LocalSearchResult: Final partition and local-search statistic.
    """
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)
    initial_score = partition_density(G, partition)

    move_count = 0
    used_passes = 0

    for _ in range(max_passes):
        used_passes += 1
        improved = False

        nodes = list(G.nodes())
        rng.shuffle(nodes)

        for node in nodes:
            candidate = best_move_for_node(state, node)
            if candidate is None or candidate.delta <= epsilon:
                continue

            apply_move_candidate(state, candidate)
            move_count += 1
            improved = True
            break

        if not improved:
            break

    return _result(G, state, initial_score, move_count, used_passes)


def refine_partition_move_best(
        G: nx.Graph,
        partition: Partition,
        max_passes: int = 1000,
        random_seed: Optional[int] = None,
        epsilon: float = 1e-12
) -> LocalSearchResult:
    """
    Refines a partition using best-improvement node moves.

    Args:
        G (nx.Graph): Input graph.
        partition (Partition): Initial partition.
        max_passes (int): Maximum number of passes over all nodes.
        random_seed (Optional[int]): Random seed for node order shuffling.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        LocalSearchResult: Final partition and local-search statistic.
    """
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)
    initial_score = partition_density(G, partition)

    move_count = 0
    used_passes = 0

    for _ in range(max_passes):
        used_passes += 1

        nodes = list(G.nodes())
        rng.shuffle(nodes)

        best: MoveCandidate | None = None

        for node in nodes:
            candidate = best_move_for_node(state, node)
            if candidate is None:
                continue

            if best is None or candidate.delta > best.delta:
                best = candidate

        if best is None or best.delta <= epsilon:
            break


        apply_move_candidate(state, best)
        move_count += 1

    return _result(G, state, initial_score, move_count, used_passes)


def refine_partition_move_plateau(
        G: nx.Graph,
        partition: Partition,
        max_passes: int = 1000,
        random_seed: Optional[int] = None,
        epsilon: float = 1e-12
) -> LocalSearchResult:
    """
    Refines a partition using node moves with bounded plateau walking.
    In addition to strictly improving moves, zero-gain moves are also accepted.

    Args:
        G (nx.Graph): Input graph.
        partition (Partition): Initial partition.
        max_passes (int): Maximum number of passes over all nodes.
        random_seed (Optional[int]): Random seed for node order shuffling.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        LocalSearchResult: Final partition and local-search statistic.
    """
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)
    initial_score = partition_density(G, partition)

    move_count = 0
    used_passes = 0
    zero_gain_count = 0
    zero_gain_limit = 2 * G.number_of_nodes()

    for _ in range(max_passes):
        used_passes += 1
        changed = False

        nodes = list(G.nodes())
        rng.shuffle(nodes)

        for node in nodes:
            candidate = best_move_for_node(state, node)
            if candidate is None:
                continue

            is_improving = candidate.delta > epsilon
            is_zero_gain = abs(candidate.delta) <= epsilon

            if not is_improving:
                if not is_zero_gain:
                    continue
                if zero_gain_count >= zero_gain_limit:
                    continue

            apply_move_candidate(state, candidate)
            move_count += 1
            changed = True

            if is_zero_gain:
                zero_gain_count += 1

        if not changed:
            break

    return _result(G, state, initial_score, move_count, used_passes)