import networkx as nx

from dense_graph_partition.adapters.kapoce import nx_to_kapoce_instance, parse_kapoce_edits, apply_edits, \
    cluster_graph_to_partition


def test_nx_to_kapoce_instance_uses_one_based_vertices() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])

    instance = nx_to_kapoce_instance(graph)

    assert instance == "p cep 3 2\n1 2\n2 3\n"


def test_parse_kapoce_edits_reads_edge_pairs() -> None:
    output = """
    1 2
    3 4
    """

    edits = parse_kapoce_edits(output)

    assert edits == [(0, 1), (2, 3)]


def test_apply_edits_toggles_edges() -> None:
    graph = nx.path_graph(3)

    edited = apply_edits(graph, [(0, 1), (0, 2)])

    assert not edited.has_edge(0, 1)
    assert edited.has_edge(0, 2)
    assert graph.has_edge(0, 1)


def test_cluster_graph_to_partition_uses_connected_components() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (2, 3)])
    graph.add_node(4)

    partition = cluster_graph_to_partition(graph)

    assert {frozenset(cluster) for cluster in partition} == {
        frozenset({0, 1}),
        frozenset({2, 3}),
        frozenset({4}),
    }