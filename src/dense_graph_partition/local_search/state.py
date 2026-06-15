from dataclasses import dataclass

import networkx as nx

from dense_graph_partition.core.types import Partition, Cluster, Node


@dataclass
class PartitionState:
    """
    Represents the internal state of a partition for efficient local search updates.

    Attributes:
        G (nx.Graph): Input graph.
        clusters (list[Cluster]): Current list of clusters.
        cluster_of (dict[Node, int]): Maps every node to the index of its current cluster.
        cluster_sizes (list[int]): Current size of each cluster.
        internal_edges (list[int]): Current number of internal edges for each cluster.
    """
    G: nx.Graph
    clusters: list[Cluster]
    cluster_of: dict[Node, int]
    cluster_sizes: list[int]
    internal_edges: list[int]

    def score(self) -> float:
        """
        Calculates the total density score of the current partition.

        Returns:
            float: Sum of density contributions over all non-empty clusters.
        """
        total = 0.0

        for edges, size in zip(self.internal_edges, self.cluster_sizes):
            if size > 0:
                total += edges / size

        return total


def build_partition_state(G: nx.Graph, partition: Partition) -> PartitionState:
    """
    Constructs a mutable local-search state from a graph and a partition.

    Args:
        G (nx.Graph): Input graph.
        partition (Partition): Initial partition of the graph nodes.

    Returns:
        PartitionState: State containing clusters, node-to-cluster assignments, cluster sizes, and internal edge counts.
    """
    clusters: list[Cluster] = []
    cluster_of: dict[Node, int] = {}

    for cluster_index, cluster in enumerate(partition):
        cluster_copy = set(cluster)
        clusters.append(cluster_copy)
        for node in cluster_copy:
            cluster_of[node] = cluster_index

    cluster_sizes = [len(cluster) for cluster in clusters]
    internal_edges = [G.subgraph(cluster).number_of_edges() for cluster in clusters]

    return PartitionState(G, clusters, cluster_of, cluster_sizes, internal_edges)

def state_to_partition(state: PartitionState) -> Partition:
    """
    Converts a partition state back into a regular partition.

    Args:
        state (PartitionState): Current local-search state.

    Returns:
        Partition: Partition represented as a list of non-empty node sets.
    """
    return [set(cluster) for cluster in state.clusters if cluster]

def neighbors_in_cluster(state: PartitionState, node: int, cluster_index: int) -> int:
    """
    Counts neighbors of a node inside a specific cluster.

    Args:
        state (PartitionState): Current local-search state.
        node (int): Node whose neighbors are counted.
        cluster_index (int): Target cluster index.

    Returns:
        int: The number of neighbors of ``node`` that currently belong to the target cluster.
    """
    return sum(1 for neighbor in state.G.neighbors(node) if state.cluster_of[neighbor] == cluster_index)
