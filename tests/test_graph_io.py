import json
from pathlib import Path

import networkx as nx

from dense_graph_partition.core.graph_io import save_graph_json, load_graph_json, load_instances_json


def test_save_and_load_graph_json(tmp_path: Path) -> None:
    G = nx.path_graph(4)
    G.graph["name"] = "path4"

    path = tmp_path / "path4.json"
    save_graph_json(G, path)

    loaded = load_graph_json(path)

    assert loaded.name == "path4"
    assert loaded.graph.number_of_nodes() == 4
    assert set(loaded.graph.edges()) == {(0, 1), (1, 2), (2, 3)}


def test_load_instances_sorted(tmp_path: Path) -> None:
    save_graph_json(nx.path_graph(3), tmp_path / "b.json", name="b")
    save_graph_json(nx.path_graph(4), tmp_path / "a.json", name="a")

    instances = load_instances_json(tmp_path)

    assert [G.name for G in instances] == ["a", "b"]


def test_load_graph_json_reads_instance(tmp_path: Path) -> None:
    path = tmp_path / "triangle.json"

    data = {
        "name": "triangle",
        "n": 3,
        "m": 3,
        "edges": [[0, 1], [1, 2], [0, 2]],
        "communities": [[0, 1, 2]],
    }

    path.write_text(json.dumps(data), encoding="utf-8")

    instance = load_graph_json(path)

    assert instance.name == "triangle"
    assert instance.graph.number_of_nodes() == 3
    assert instance.graph.number_of_edges() == 3
    assert instance.graph.has_edge(0, 1)
    assert instance.communities == [[0, 1, 2]]
