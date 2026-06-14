import subprocess
from pathlib import Path

import networkx as nx

from dense_graph_partition.core.types import Partition


def nx_to_kapoce_instance(G: nx.Graph) -> str:
    """
    Converts a networkx graph into a string format for the KaPoCE algorithm.

    Args:
        G (nx.Graph): The networkx graph to convert.

    Returns:
        str: The KaPoCE instance as a string.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    lines = [f"p cep {n} {m}"]
    lines.extend(f"{u + 1} {v + 1}" for u, v in G.edges())

    return "\n".join(lines) + "\n"


def parse_kapoce_edits(output: str) -> list[tuple[int, int]]:
    """
    Parse edge additions and deletions from KaPoCE's stdout output.

    Args:
        output (str): The KaPoCE output string.

    Returns:
        list[tuple[int, int]]: A list of (u, v) pairs indicating edges to toggle.
    """
    edits: list[tuple[int, int]] = []

    for line in output.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()
        if len(parts) != 2:
            continue

        u, v = map(int, parts)
        edits.append((u - 1, v - 1))

    return edits


def apply_edits(G: nx.Graph, edits: list[tuple[int, int]]) -> nx.Graph:
    """
    Applies edge additions and deletions to a graph.

    Args:
        G (nx.Graph): The networkx graph to be converted. :
        edits (list[tuple[int, int]]): A list of (u, v) edges to toggle.

    Returns:
        nx.Graph: The networkx graph with the edits applied.
    """
    H = G.copy()

    edges_to_remove = []
    edges_to_add = []

    for u, v in edits:
        if H.has_edge(u, v):
            edges_to_remove.append((u, v))
        else:
            edges_to_add.append((u, v))

    H.remove_edges_from(edges_to_remove)
    H.add_edges_from(edges_to_add)

    return H


def cluster_graph_to_partition(H: nx.Graph) -> Partition:
    """
    Extracts a partition from a cluster graph's connected components.

    Args:
        H (nx.Graph): The cluster graph (a graph consisting of disjoint cliques).

    Returns:
        Partition: The partition of the graph nodes into disjoint sets.
    """
    return [set(component) for component in nx.connected_components(H)]


def kapoce_partition(G: nx.Graph, executable_path: Path, config_path: Path) -> Partition:
    """
    Computes a partition using the external KaPoCE executable.
    KaPoCE solves the Cluster Editing Problem (CEP). The graph is sent via stdin in the format for the PACE challenge, and the resulting edge additions/deletions yield a cluster graph.

    Args:
        G (nx.Graph): The networkx graph to partition.
        executable_path (Path): Path to the KaPoCE executable.
        config_path (Path): Path to the KaPoCE configuration file.

    Returns:
        Partition: A partition derived from the resulting cluster graph.

    Raises:
        FileNotFoundError: If the executable or config file cannot be found.
    """
    if not executable_path.exists():
        raise FileNotFoundError(f"KaPoCE executable not found: {executable_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"KaPoCE config not found: {config_path}")

    input_data = nx_to_kapoce_instance(G)

    cmd = [str(executable_path), "-c", str(config_path)]

    result = subprocess.run(
        cmd, input=input_data, text=True, capture_output=True, cwd=executable_path.parent, timeout=300
    )

    if result.returncode != 0:
        raise RuntimeError("KaPoCE failed with exit code "f"{result.returncode}.\nSTDERR:\n{result.stderr}")

    edits = parse_kapoce_edits(result.stdout)
    edited_graph = apply_edits(G, edits)

    return cluster_graph_to_partition(edited_graph)