import networkx as nx

from dense_graph_partition.core.types import Partition


def _partition_from_matching(G: nx.Graph, matching: set[tuple[int, int]]) -> Partition:
    """
    Builds a partition from a matching.
    Matched edges form a cluster of size two. All unmatched nodes are placed into singleton clusters.

    Args:
        G (nx.Graph): The input graph.
        matching (set[tuple[int, int]]): A set of independent edges.

    Returns:
        Partition: A partition induced by the matching.
    """
    partition: Partition = []
    used_nodes = set()

    for u, v in matching:
        partition.append({u, v})
        used_nodes.add(u)
        used_nodes.add(v)

    for node in G.nodes:
        if node not in used_nodes:
            partition.append({node})

    return partition


def matching_partition(G: nx.Graph) -> Partition:
    """
    Computes a partition from a maximal matching.
    This is a fast greedy baseline, where the matching is maximal but not necessarily maximum.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        Partition: A partition induced by the maximal matching.
    """
    matching = nx.maximal_matching(G)
    return _partition_from_matching(G, matching)


def maximum_matching_partition(G: nx.Graph) -> Partition:
    """
    Computes a partition from a maximum cardinality matching.
    This baseline maximizes the number of matched edges. This gives the standard matching-based 2-approximation start partition.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        Partition: A partition induced by the maximum cardinality matching.
    """
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return _partition_from_matching(G, matching)


def high_degree_first_matching_partition(G: nx.Graph) -> Partition:
    """
    Computes a partition from a greedy matching using high-degree edges first.
    This matching favors edges incident to high-degree nodes.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        Partition: A partition induced by the greedy high-degree-first matching.
    """
    edges = list(G.edges())
    edges.sort(
        key=lambda e: (
            max(G.degree(e[0]), G.degree(e[1])),
            min(G.degree(e[0]), G.degree(e[1])),
        ),
        reverse=True
    )

    matching: set[tuple[int, int]] = set()
    used_nodes = set()

    for u, v in edges:
        if u in used_nodes or v in used_nodes:
            continue

        matching.add((u,v))
        used_nodes.add(u)
        used_nodes.add(v)

    return _partition_from_matching(G, matching)


def high_degree_product_matching_partition(G: nx.Graph) -> Partition:
    """
    Computes a partition from a greedy matching using the degree product of edges.
    This matching favors edges connecting two high-degree nodes.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        Partition: A partition induced by the greedy high-degree-product matching.
    """
    edges = list(G.edges())
    edges.sort(key=lambda edge: G.degree(edge[0]) * G.degree(edge[1]), reverse=True)

    matching: set[tuple[int, int]] = set()
    used_nodes = set()

    for u, v in edges:
        if u in used_nodes or v in used_nodes:
            continue

        matching.add((u,v))
        used_nodes.add(u)
        used_nodes.add(v)

    return _partition_from_matching(G, matching)

def maximum_matching_edge_cover_partition(G: nx.Graph) -> Partition:
    """
    Computes a star-like partition based on a maximum cardinality matching.
    Each matching edge initially forms a cluster. Afterwards, every unmatched vertex is greedily added to the cluster of one of its matched neighbors.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        Partition: A partition consisting of matching-edge clusters extended by unmatched neighboring nodes.
    """
    matching = nx.max_weight_matching(G, maxcardinality=True)

    partition: Partition = []
    node_to_cluster: dict[int, set[int]] = {}
    matched_nodes: set[int] = set()

    for u, v in matching:
        cluster = {u, v}
        partition.append(cluster)

        node_to_cluster[u] = cluster
        node_to_cluster[v] = cluster

        matched_nodes.add(u)
        matched_nodes.add(v)

    for node in G.nodes:
        if node in matched_nodes:
            continue

        matched_neighbor = next(
            neighbor
            for neighbor in G.neighbors(node)
            if neighbor in matched_nodes
        )

        cluster = node_to_cluster[matched_neighbor]
        cluster.add(node)
        node_to_cluster[node] = cluster

    return partition