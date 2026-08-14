import networkx as nx

from dense_graph_partition.core.types import Partition


def singleton_partition(G: nx.Graph) -> Partition:
    """
    Builds the singleton partition.

    Args:
        G (nx.Graph): Input graph.

    Returns:
        Partition: Partition where every node forms its own cluster.
    """
    return [{node} for node in G.nodes()]