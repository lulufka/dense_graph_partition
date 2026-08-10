from __future__ import annotations

import networkx as nx

from dense_graph_partition.algorithms.matching import maximum_matching_partition
from dense_graph_partition.core.evaluation import partition_density
from dense_graph_partition.core.types import Partition
from dense_graph_partition.local_search.pipeline import run_local_search_pipeline


def copy_partition(partition: Partition) -> Partition:
    """
    Creates a deep copy of a partition.

    Args:
        partition (Partition): Partition to copy.

    Returns:
        Partition: Independent copy.
    """
    return [set(cluster) for cluster in partition]


def mdgp_plateau_partition(G: nx.Graph, runs: int = 15, base_seed: int = 42) -> Partition:
    """
    Computes a partition using the final MDGP-Plateau heuristic.

    Args:
        G (nx.Graph): The networkx graph to partition.
        runs (int): Number of independent randomized runs.
        base_seed (int): Base seed used for reproducible runs.

    Returns:
        Partition: The best partition found across all runs.
    """
    if G.number_of_nodes() == 0:
        return []

    start_partition = maximum_matching_partition(G)

    pipeline = ",".join(["move_plateau"] * 4)

    best_partition: Partition | None = None
    best_density = float("-inf")

    for run_index in range(runs):
        seed = base_seed + 1_000_000 * run_index

        result = run_local_search_pipeline(
            G=G,
            partition=copy_partition(start_partition),
            pipeline=pipeline,
            zero_gain_factor=4,
            random_seed=seed,
        )

        density = partition_density(G, result.partition)

        if density > best_density:
            best_density = density
            best_partition = result.partition

    if best_partition is None:
        raise RuntimeError("MDGP-Plateau did not produce a partition.")

    return best_partition