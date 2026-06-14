import leidenalg
import networkx as nx
from pytest import mark

from dense_graph_partition.adapters.leiden import nx_to_igraph, membership_to_partition, leiden_mdgp_partition


def test_nx_to_igraph_preserves_node_mapping() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(10, 20), (20, 30)])

    ig_graph, node_order = nx_to_igraph(graph)

    assert ig_graph.vcount() == 3
    assert ig_graph.ecount() == 2
    assert node_order == [10, 20, 30]


def test_membership_to_partition() -> None:
    membership = [0, 1, 0, 1]
    node_order = [10, 20, 30, 40]

    partition = membership_to_partition(membership, node_order)

    assert {frozenset(cluster) for cluster in partition} == {
        frozenset({10, 30}),
        frozenset({20, 40}),
    }


@mark.skipif(
    not hasattr(leidenalg, "MDGPVertexPartition"),
    reason="Custom MDGPVertexPartition is not available.",
)
def test_leiden_mdgp_partition_returns_valid_partition() -> None:
    graph = nx.cycle_graph(4)

    partition = leiden_mdgp_partition(graph)

    assert set().union(*partition) == set(graph.nodes())