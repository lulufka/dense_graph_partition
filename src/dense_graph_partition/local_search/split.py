from dataclasses import dataclass

import networkx as nx

from dense_graph_partition.core.evaluation import partition_density
from dense_graph_partition.core.types import Cluster, Partition
from dense_graph_partition.local_search.result import LocalSearchResult
from dense_graph_partition.local_search.search import build_local_search_result
from dense_graph_partition.local_search.state import PartitionState, build_partition_state


@dataclass(frozen=True)
class SplitCandidate:
    """
    Represents a candidate cluster split.

    Attributes:
        cluster_index (int): Index of the cluster to split.
        first_cluster (Cluster): First part of the split.
        second_cluster (Cluster): Second part of the split.
        delta (float): Expected change in density.
    """
    cluster_index: int
    first_cluster: Cluster
    second_cluster: Cluster
    delta: float


def split_delta(state: PartitionState, cluster_index: int, first_cluster: Cluster, second_cluster: Cluster) -> float:
    """
    Computes the density change of replacing one cluster by two clusters.

    Args:
        state (PartitionState): Current local-search state.
        cluster_index (int): Index of the cluster to split.
        first_cluster (Cluster): First proposed split part.
        second_cluster (Cluster): Second proposed split part.

    Returns:
        float: Change in density. Positive values are improvements.
    """
    old_size = state.cluster_sizes[cluster_index]
    old_edges = state.internal_edges[cluster_index]

    if old_size == 0:
        return float("-inf")

    first_size = len(first_cluster)
    second_size = len(second_cluster)

    if first_size == 0 or second_size == 0:
        return float("-inf")

    first_edges = state.G.subgraph(first_cluster).number_of_edges()
    second_edges = state.G.subgraph(second_cluster).number_of_edges()

    old_score = old_edges / old_size
    new_score = (first_edges / first_size) + (second_edges / second_size)

    return new_score - old_score


def apply_split(state: PartitionState, cluster_index: int, first_cluster: Cluster, second_cluster: Cluster) -> None:
    """
    Splits one cluster into two new clusters. The original cluster is left as an empty slot.

    Args:
        state (PartitionState): Current local-search state.
        cluster_index (int): Index of the cluster to split.
        first_cluster (Cluster): First split part.
        second_cluster (Cluster): Second split part.

    Raises:
        ValueError: If the source cluster is empty, if the split parts do not cover the original cluster, if the splits overlap, or if one split part is empty.
    """
    if state.cluster_sizes[cluster_index] == 0:
        raise ValueError("Cannot split an empty cluster.")

    original_cluster = state.clusters[cluster_index]

    if first_cluster | second_cluster != original_cluster:
        raise ValueError("Split parts do not cover the original cluster.")

    if first_cluster & second_cluster:
        raise ValueError("Split parts must be disjoint.")

    if not first_cluster or not second_cluster:
        raise ValueError("Split parts must not be empty.")

    state.clusters[cluster_index].clear()
    state.cluster_sizes[cluster_index] = 0
    state.internal_edges[cluster_index] = 0

    first_index = len(state.clusters)
    second_index = first_index + 1

    first_copy = set(first_cluster)
    second_copy = set(second_cluster)

    state.clusters.append(first_copy)
    state.cluster_sizes.append(len(first_copy))
    state.internal_edges.append(state.G.subgraph(first_copy).number_of_edges())
    for node in first_copy:
        state.cluster_of[node] = first_index

    state.clusters.append(second_copy)
    state.cluster_sizes.append(len(second_copy))
    state.internal_edges.append(state.G.subgraph(second_copy).number_of_edges())
    for node in second_copy:
        state.cluster_of[node] = second_index


def apply_split_candidate(state: PartitionState, candidate: SplitCandidate) -> None:
    """
    Applies a split candidate to the current state.

    Args:
        state (PartitionState): Current local-search state.
        candidate (SplitCandidate): Candidate split to apply.
    """
    apply_split(state, candidate.cluster_index, candidate.first_cluster, candidate.second_cluster)


def min_cut_split_candidate(state: PartitionState, cluster_index: int) -> SplitCandidate | None:
    """
    Computes a split candidate using the minimum cut algorithm.

    Args:
        state (PartitionState): Current local-search state.
        cluster_index (int): Index of the cluster to split.

    Returns:
        SplitCandidate | None: Candidate split, or ``None`` if no valid split can be computed.
    """
    cluster = state.clusters[cluster_index]
    if len(cluster) < 4:
        return None

    subgraph = state.G.subgraph(cluster)
    if not nx.is_connected(subgraph):
        return None

    try:
        _, parts = nx.stoer_wagner(subgraph)
    except nx.NetworkXException:
        return None

    first_cluster = set(parts[0])
    second_cluster = set(parts[1])

    delta = split_delta(state, cluster_index, first_cluster, second_cluster)

    return SplitCandidate(cluster_index, first_cluster, second_cluster, delta)


def bridge_split_candidate(state: PartitionState, cluster_index: int) -> SplitCandidate | None:
    """
    Computes a split candidate by removing a bridge inside a cluster.

    Args:
        state (PartitionState): Current local-search state.
        cluster_index (int): Index of the cluster to split.

    Returns:
        SplitCandidate | None: Best bridge-based split candidate, or ``None`` if the cluster has no valid bridge split.
    """
    cluster = state.clusters[cluster_index]
    if len(cluster) < 4:
        return None

    subgraph = state.G.subgraph(cluster)
    if not nx.is_connected(subgraph):
        return None

    best: SplitCandidate | None = None

    for u, v in nx.bridges(subgraph):
        candidate_graph = subgraph.copy()
        candidate_graph.remove_edge(u, v)

        components = list(nx.connected_components(candidate_graph))
        if len(components) != 2:
            continue

        first_cluster = set(components[0])
        second_cluster = set(components[1])

        delta = split_delta(state, cluster_index, first_cluster, second_cluster)

        candidate = SplitCandidate(cluster_index, first_cluster, second_cluster, delta)

        if best is None or candidate.delta > best.delta:
            best = candidate

    return best


def best_min_cut_split(state: PartitionState, epsilon: float = 1e-12) -> SplitCandidate | None:
    """
    Finds the best improving minimum-cut split over all clusters.

    Args:
        state (PartitionState): Current local-search state.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        SplitCandidate | None: Best improving minimum-cut split, or ``None`` if no improving split exists.
    """
    best: SplitCandidate | None = None

    for cluster_index, cluster in enumerate(state.clusters):
        if not cluster:
            continue

        candidate = min_cut_split_candidate(state, cluster_index)
        if candidate is None or candidate.delta <= epsilon:
            continue

        if best is None or candidate.delta > best.delta:
            best = candidate

    return best


def best_bridge_split(state: PartitionState, epsilon: float = 1e-12) -> SplitCandidate | None:
    """
    Finds the best improving bridge split over all clusters.

    Args:
        state (PartitionState): Current local-search state.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        SplitCandidate | None: Best improving bridge split, or ``None`` if no improving bridge split exists.
    """
    best: SplitCandidate | None = None

    for cluster_index, cluster in enumerate(state.clusters):
        if not cluster:
            continue

        candidate = bridge_split_candidate(state, cluster_index)
        if candidate is None or candidate.delta <= epsilon:
            continue

        if best is None or candidate.delta > best.delta:
            best = candidate

    return best


def refine_partition_split_min_cut(G: nx.Graph, partition: Partition, max_passes: int = 1000, epsilon: float = 1e-12) -> LocalSearchResult:
    """
    Refines a partition using repeated minimum-cut cluster splits.

    Args:
        G (nx.Graph): Input Graph.
        partition (Partition): Initial partition.
        max_passes (int): Maximum number of split iterations.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        LocalSearchResult: Final partition and local search statistics.
    """
    state = build_partition_state(G, partition)
    initial_score = partition_density(G, partition)

    split_count = 0
    used_passes = 0

    for _ in range(max_passes):
        used_passes += 1

        candidate = best_min_cut_split(state, epsilon)

        if candidate is None:
            break

        apply_split_candidate(state, candidate)
        split_count += 1

    return build_local_search_result(G, state, initial_score, split_count, used_passes)


def refine_partition_bridge_split(G: nx.Graph, partition: Partition, max_passes: int = 1000, epsilon: float = 1e-12) -> LocalSearchResult:
    """
    Refines a partition using repeated bridge-based cluster splits.

    Args:
        G (nx.Graph): Input Graph.
        partition (Partition): Initial partition.
        max_passes (int): Maximum number of split iterations.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        LocalSearchResult: Final partition and local search statistics.
    """
    state = build_partition_state(G, partition)
    initial_score = partition_density(G, partition)

    split_count = 0
    used_passes = 0

    for _ in range(max_passes):
        used_passes += 1

        candidate = best_bridge_split(state, epsilon)

        if candidate is None:
            break

        apply_split_candidate(state, candidate)
        split_count += 1

    return build_local_search_result(G, state, initial_score, split_count, used_passes)