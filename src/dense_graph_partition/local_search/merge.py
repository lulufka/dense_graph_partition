import random
from dataclasses import dataclass

import networkx as nx

from dense_graph_partition.core.evaluation import partition_density
from dense_graph_partition.core.types import Partition
from dense_graph_partition.local_search.result import LocalSearchResult, build_local_search_result
from dense_graph_partition.local_search.state import PartitionState, build_partition_state


@dataclass(frozen=True)
class MergeCandidate:
    """
    Represents a candidate cluster merge.

    Attributes:
        cluster_a (int): Index of the first cluster.
        cluster_b (int): Index of the second cluster.
        delta (float): Expected change in density. Positive values are improvements.
    """
    cluster_a: int
    cluster_b: int
    delta: float


def intercluster_edges(state: PartitionState, cluster_a: int, cluster_b: int) -> int:
    """
    Count edges between two clusters.

    Args:
        state (PartitionState): Current local-search state.
        cluster_a (int): Index of the first cluster.
        cluster_b (int): Index of the second cluster.

    Returns:
        int: Number of edges with one endpoint in each cluster.
    """
    if cluster_a == cluster_b:
        return 0

    nodes_a = state.clusters[cluster_a]
    nodes_b = state.clusters[cluster_b]

    if not nodes_a or not nodes_b:
        return 0

    if len(nodes_a) > len(nodes_b):
        nodes_a, nodes_b = nodes_b, nodes_a

    nodes_b_set = set(nodes_b)

    count = 0
    for node in nodes_a:
        for neighbor in state.G.neighbors(node):
            if neighbor in nodes_b_set:
                count += 1

    return count


def delta_merge_clusters(state: PartitionState, cluster_a: int, cluster_b: int) -> float:
    """
    Computes the density score change when merging two clusters.

    Args:
        state (PartitionState): Current local-search state.
        cluster_a (int): Index of the first cluster.
        cluster_b (int): Index of the second cluster.

    Returns:
        float: Change in density. Positive values are improvements.
    """
    if cluster_a == cluster_b:
        return float("-inf")

    size_a = state.cluster_sizes[cluster_a]
    size_b = state.cluster_sizes[cluster_b]

    if size_a == 0 or size_b == 0:
        return float("-inf")

    edges_a = state.internal_edges[cluster_a]
    edges_b = state.internal_edges[cluster_b]
    edges_between = intercluster_edges(state, cluster_a, cluster_b)

    old_score = (edges_a / size_a) + (edges_b / size_b)
    new_score = (edges_a + edges_b + edges_between) / (size_a + size_b)

    return new_score - old_score


def apply_merge_clusters(state: PartitionState, cluster_a: int, cluster_b: int) -> None:
    """
    Merges two clusters and updates the partition state in place. The second cluster is left as an empty slot.

    Args:
        state (PartitionState): Current local-search state.
        cluster_a (int): Index of the first cluster.
        cluster_b (int): Index of the second cluster.

    Raises:
        ValueError: If both indices refer to the same cluster or if one cluster is empty.
    """
    if cluster_a == cluster_b:
        raise ValueError("Cannot merge a cluster with itself.")

    size_a = state.cluster_sizes[cluster_a]
    size_b = state.cluster_sizes[cluster_b]

    if size_a == 0 or size_b == 0:
        raise ValueError("Cannot merge empty clusters.")

    edges_between = intercluster_edges(state, cluster_a, cluster_b)

    for node in state.clusters[cluster_b]:
        state.cluster_of[node] = cluster_a

    state.clusters[cluster_a].update(state.clusters[cluster_b])
    state.cluster_sizes[cluster_a] = size_a + size_b
    state.internal_edges[cluster_a] = state.internal_edges[cluster_a] + state.internal_edges[cluster_b] + edges_between

    state.clusters[cluster_b].clear()
    state.cluster_sizes[cluster_b] = 0
    state.internal_edges[cluster_b] = 0


def apply_merge_candidate(state: PartitionState, candidate: MergeCandidate) -> None:
    """
    Applies a merge candidate to the current state.

    Args:
        state (PartitionState): Current local-search state.
        candidate (MergeCandidate): Merge candidate to apply.
    """
    apply_merge_clusters(state, candidate.cluster_a, candidate.cluster_b)


def neighboring_cluster_pairs(state: PartitionState) -> list[tuple[int, int]]:
    """
    Returns all cluster pairs that are connected by at least one edge.

    Args:
        state (PartitionState): Current local-search state.

    Returns:
        list[tuple[int, int]]: Sorted list of neighboring cluster index pairs.
    """
    pairs: set[tuple[int, int]] = set()

    for u, v in state.G.edges():
        cluster_u = state.cluster_of[u]
        cluster_v = state.cluster_of[v]

        if cluster_u == cluster_v:
            continue

        if state.cluster_sizes[cluster_u] == 0 or state.cluster_sizes[cluster_v] == 0:
            continue

        if cluster_u > cluster_v:
            cluster_u, cluster_v = cluster_v, cluster_u

        pairs.add((cluster_u, cluster_v))

    return sorted(pairs)


def best_merge_pair(state: PartitionState, epsilon: float = 1e-12) -> MergeCandidate | None:
    """
    Finds the best improving merge among neighboring cluster pairs.

    Args:
        state (PartitionState): Current local-search state.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        MergeCandidate | None: Best improving merge candidate, or ``None`` if no improving merge exists.
    """
    best: MergeCandidate | None = None

    for cluster_a, cluster_b in neighboring_cluster_pairs(state):
        delta = delta_merge_clusters(state, cluster_a, cluster_b)

        if delta <= epsilon:
            continue

        candidate = MergeCandidate(cluster_a, cluster_b, delta)

        if best is None or candidate.delta > best.delta:
            best = candidate

    return best


def first_improving_merge_pair(state: PartitionState, pairs: list[tuple[int, int]], epsilon: float = 1e-12) -> MergeCandidate | None:
    """
    Finds the first improving merge among neighboring cluster pairs.

    Args:
        state (PartitionState): Current local-search state.
        pairs (list[tuple[int, int]]): Candidate cluster pairs.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        MergeCandidate | None: First improving merge, or ``None`` if no improving merge exists.

    """
    for cluster_a, cluster_b in pairs:
        delta = delta_merge_clusters(state, cluster_a, cluster_b)

        if delta > epsilon:
            return MergeCandidate(cluster_a, cluster_b, delta)

    return None


def refine_partition_merge_first(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 1000,
    random_seed: int | None = None,
    epsilon: float = 1e-12,
) -> LocalSearchResult:
    """
    Refines a partition using first-improvement cluster merges.

    Args:
        G (nx.Graph): Input graph.
        partition (Partition): Initial partition.
        max_passes (int): Maximum number of merge iterations.
        random_seed (int | None): Random seed for candidate pair shuffling.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        LocalSearchResult: Final partition and local-search statistics.
    """
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)
    initial_score = partition_density(G, partition)

    merge_count = 0
    used_passes = 0

    for _ in range(max_passes):
        used_passes += 1

        pairs = neighboring_cluster_pairs(state)
        rng.shuffle(pairs)

        candidate = first_improving_merge_pair(state, pairs, epsilon)

        if candidate is None:
            break

        apply_merge_candidate(state, candidate)
        merge_count += 1

    return build_local_search_result(G, state, initial_score, merge_count, used_passes)


def refine_partition_merge_best(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 1000,
    epsilon: float = 1e-12,
) -> LocalSearchResult:
    """
    Refines a partition using best-improvement cluster merges.

    Args:
        G (nx.Graph): Input graph.
        partition (Partition): Initial partition.
        max_passes (int): Maximum number of merge iterations.
        epsilon (float): Numerical tolerance for improvement checks.

    Returns:
        LocalSearchResult: Final partition and local-search statistics.
    """
    state = build_partition_state(G, partition)
    initial_score = partition_density(G, partition)

    merge_count = 0
    used_passes = 0

    for _ in range(max_passes):
        used_passes += 1

        candidate = best_merge_pair(state, epsilon)

        if candidate is None:
            break

        apply_merge_candidate(state, candidate)
        merge_count += 1

    return build_local_search_result(G, state, initial_score, merge_count, used_passes)
