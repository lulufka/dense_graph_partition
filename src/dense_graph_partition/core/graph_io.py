import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass(frozen=True)
class GraphInstance:
    name: str
    graph: nx.Graph
    communities: list[list[int]]


def load_graph_json(path: Path) -> GraphInstance:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    graph = nx.Graph()
    graph.add_nodes_from(range(data["n"]))
    graph.add_edges_from(data["edges"])

    return GraphInstance(
        name=data["name"],
        graph=graph,
        communities=data.get("communities", []),
    )


def save_graph_json(G: nx.Graph, path: Path, name: str | None = None, communities: list[list[int]] | None = None) -> None:
    data = {
        "name": name or G.graph.get("name", path.stem),
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "edges": [list(edge) for edge in G.edges()],
        "communities": communities or []
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def load_instances_json(data_dir: Path) -> list[GraphInstance]:
    paths = sorted(data_dir.glob("*.json"))
    return [load_graph_json(path) for path in paths]
