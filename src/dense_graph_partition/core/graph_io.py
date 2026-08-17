import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from dense_graph_partition.core.evaluation import validate_partition
from dense_graph_partition.core.types import Partition


@dataclass(frozen=True)
class GraphInstance:
    name: str
    graph: nx.Graph


@dataclass(frozen=True)
class GroundTruthGraphInstance(GraphInstance):
    ground_truth: Partition
    metadata: dict[str, object]


def load_graph_json(path: Path) -> GraphInstance:
    """
    Loads a graph instance from a JSON file.

    Args:
        path (Path): Path to the JSON instance file.

    Returns:
        GraphInstance: Loaded graph instance.
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    graph = nx.Graph()
    graph.add_nodes_from(range(data["n"]))
    graph.add_edges_from(data["edges"])

    return GraphInstance(
        name=data["name"],
        graph=graph,
    )


def load_ground_truth_graph_json(path: Path) -> GroundTruthGraphInstance:
    """
    Loads a graph instance including its ground-truth partition.

    Args:
        path (Path): Path to the JSON instance file.

    Returns:
        GroundTruthGraphInstance: Loaded graph instance with ground truth.
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    graph = nx.Graph()
    graph.add_nodes_from(range(data["n"]))
    graph.add_edges_from(data["edges"])

    ground_truth = [set(community) for community in data["ground_truth"]]

    validate_partition(graph, ground_truth)

    return GroundTruthGraphInstance(
        name=data["name"],
        graph=graph,
        ground_truth=ground_truth,
        metadata=data.get("metadata", {}),
    )


def save_graph_json(G: nx.Graph, path: Path, name: str | None = None) -> None:
    """
    Saves a graph instance as JSON.
    """
    data = {
        "name": name or G.graph.get("name", path.stem),
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "edges": [list(edge) for edge in G.edges()],
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def save_ground_truth_graph_json(G: nx.Graph, path: Path, name: str, ground_truth: Partition,  metadata: dict[str, object] | None = None) -> None:
    """
    Saves a graph instance including its ground-truth partition.

    Args:
        G (nx.Graph): Graph to save.
        path (Path): Output path.
        name (str): Instance name.
        ground_truth (Partition): Ground-truth partition.
        metadata (dict[str, object] | None): Additional generator metadata.
    """
    data = {
        "name": name,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "edges": [list(edge) for edge in G.edges()],
        "ground_truth": [sorted(cluster) for cluster in ground_truth],
        "metadata": metadata or {},
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def load_instances_json(data_dir: Path) -> list[GraphInstance]:
    """
    Loads all graph instances from a directory.
    """
    paths = sorted(data_dir.glob("*.json"))
    return [load_graph_json(path) for path in paths]


def partition_to_json(partition: Partition) -> str:
    """
    Converts a partition into a JSON string.

    Args:
        partition (Partition): Partition to serialize.

    Returns:
        str: JSON representation of the partition.
    """
    partition_data = [sorted(cluster) for cluster in partition]

    return json.dumps(partition_data, separators=(",", ":"))
