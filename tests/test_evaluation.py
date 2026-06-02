import networkx as nx
import pytest

from dense_graph_partition.core.evaluation import cluster_density, partition_density, validate_partition


def test_cluster_density_for_triangle() -> None:
    G = nx.complete_graph(3)

    assert cluster_density(G, {0, 1, 2}) == 1.0


def test_partition_density_for_two_edges() -> None:
    G = nx.path_graph(4)
    partition = [{0, 1}, {2, 3}]

    assert partition_density(G, partition) == 1.0


def test_validate_partition_rejects_missing_node() -> None:
    G = nx.path_graph(3)

    with pytest.raises(ValueError, match="missing node"):
        validate_partition(G, [{0, 1}])


def test_validate_partition_rejects_duplicate_node() -> None:
    G = nx.path_graph(3)

    with pytest.raises(ValueError, match="multiple clusters"):
        validate_partition(G, [{0, 1}, {1, 2}])


def test_validate_partition_rejects_unknown_node() -> None:
    G = nx.path_graph(3)

    with pytest.raises(ValueError, match="unknown node"):
        validate_partition(G, [{0, 1, 2, 99}])