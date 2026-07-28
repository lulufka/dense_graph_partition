import networkx as nx

from dense_graph_partition.algorithms.matching import matching_partition, maximum_matching_partition, high_degree_first_matching_partition, high_degree_product_matching_partition, maximum_matching_edge_cover_partition
from dense_graph_partition.core.evaluation import validate_partition, partition_density


def test_matching_partition_is_valid() -> None:
    G = nx.path_graph(5)
    partition = matching_partition(G)

    validate_partition(G, partition)


def test_maximum_matching_partition_is_valid() -> None:
    G = nx.path_graph(5)
    partition = maximum_matching_partition(G)

    validate_partition(G, partition)


def test_high_degree_first_matching_partition_is_valid() -> None:
    G = nx.path_graph(5)
    partition = high_degree_first_matching_partition(G)

    validate_partition(G, partition)


def test_high_degree_product_matching_partition_is_valid() -> None:
    G = nx.path_graph(5)
    partition = high_degree_product_matching_partition(G)

    validate_partition(G, partition)


def test_maximum_matching_on_path4_has_two_edges() -> None:
    G = nx.path_graph(4)
    partition = maximum_matching_partition(G)

    assert len(partition) == 2
    assert partition_density(G, partition) == 1.0


def test_edge_cover_on_path4_has_two_clusters() -> None:
    G = nx.path_graph(4)

    partition = maximum_matching_edge_cover_partition(G)

    assert len(partition) == 2
    assert all(len(cluster) == 2 for cluster in partition)
    assert partition_density(G, partition) == 1.0
    validate_partition(G, partition)


def test_edge_cover_attaches_unmatched_nodes_to_matching_cluster() -> None:
    G = nx.star_graph(3)

    partition = maximum_matching_edge_cover_partition(G)

    assert partition == [{0, 1, 2, 3}]
    validate_partition(G, partition)


def test_matching_partition_keeps_isolated_nodes() -> None:
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1)

    for algorithm in [
        matching_partition,
        maximum_matching_partition,
        high_degree_product_matching_partition,
        high_degree_first_matching_partition
    ]:
        partition = algorithm(G)

        assert {2} in partition
        validate_partition(G, partition)