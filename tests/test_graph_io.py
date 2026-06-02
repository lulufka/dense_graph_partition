from pathlib import Path

import networkx as nx

from dense_graph_partition.core.graph_io import save_graph_json, load_graph_json, load_instances_json


def test_save_and_load_graph_json(tmp_path: Path) -> None:
    G = nx.path_graph(4)
    G.graph["name"] = "path4"

    path = tmp_path / "path4.json"
    save_graph_json(G, path)

    loaded = load_graph_json(path)

    assert loaded.graph["name"] == "path4"
    assert loaded.number_of_nodes() == 4
    assert set(loaded.edges()) == {(0, 1), (1, 2), (2, 3)}


def test_load_instances_sorted(tmp_path: Path) -> None:
    save_graph_json(nx.path_graph(3), tmp_path / "b.json", name="b")
    save_graph_json(nx.path_graph(4), tmp_path / "a.json", name="a")

    instances = load_instances_json(tmp_path)

    assert [G.graph["name"] for G in instances] == ["a", "b"]