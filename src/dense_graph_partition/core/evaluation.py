import networkx as nx

from dense_graph_partition.core.types import Partition, Cluster


def validate_partition(G: nx.Graph, partition: Partition) -> None:
    graph_nodes = set(G.nodes())
    seen_nodes = set()

    for cluster in partition:
        if not cluster:
            raise ValueError("Partition contains an empty cluster.")

        if seen_nodes & cluster:
            raise ValueError("At least one node appears in multiple clusters.")

        seen_nodes.update(cluster)

    missing_nodes = graph_nodes - seen_nodes
    extra_nodes = seen_nodes - graph_nodes

    if missing_nodes:
        raise ValueError("Partition has at least one missing node.")
    if extra_nodes:
        raise ValueError("Partition contains at least one unknown node.")


def cluster_density(G: nx.Graph, cluster: Cluster) -> float:
    """
    Calculates the density of a single cluster in the graph.
    The density is defined as the number of edges in the subgraph induced by the cluster, divided by the number of nodes in the cluster.

    Args:
        G (nx.Graph): The networkx graph.
        cluster (Cluster): A set of node indices representing a cluster.

    Returns:
        float: The density of the cluster.
    """
    return G.subgraph(cluster).number_of_edges() / len(cluster)


def partition_density(G: nx.Graph, partition: Partition) -> float:
    """
    Calculates the total density of a partition.
    The partition density is the sum of the densities of all its clusters.

    Args:
        G (nx.Graph): The networkx graph.
        partition (Partition): A list of sets of integers representing node clusters.

    Returns:
        float: The total density of the partition.
    """
    validate_partition(G, partition)
    return sum(cluster_density(G, cluster) for cluster in partition)


def edge_density(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    return 2 * m / (n * (n - 1))


def partition_cluster_sizes(partition: Partition) -> list[int]:
    return sorted((len(cluster) for cluster in partition), reverse=True)


def partition_num_clusters(partition: Partition) -> int:
    return len(partition)

