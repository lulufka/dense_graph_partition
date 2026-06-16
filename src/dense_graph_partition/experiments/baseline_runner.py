import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx
import pandas as pd

from dense_graph_partition.core.evaluation import validate_partition, partition_cluster_sizes, partition_density, \
    partition_num_clusters, edge_density
from dense_graph_partition.core.graph_io import load_instances_json
from dense_graph_partition.core.types import Partition


@dataclass(frozen=True)
class AlgorithmSpec:
    """
    Describes one partitioning algorithm used in an experiment.

    Attributes:
        name (str): Human-readable algorithm name used in result tables.
        run (Callable[[nx.Graph], Partition]): Function that computes a partition for a graph.
    """
    name: str
    run: Callable[[nx.Graph], Partition]


def run_algorithm(G: nx.Graph, algorithm: AlgorithmSpec) -> dict[str, object]:
    """
    Executes a partitioning algorithm on a graph and computes evaluation statistics.

    Args:
        G (nx.Graph): Input graph.
        algorithm (AlgorithmSpec): Algorithm to execute.

    Returns:
        dict[str, object]: A dictionary containing evaluation metrics and runtime information.
    """
    start = time.perf_counter()
    partition = algorithm.run(G)
    runtime_seconds = time.perf_counter() - start

    validate_partition(G, partition)

    sizes = partition_cluster_sizes(partition)

    return {
        "algorithm": algorithm.name,
        "density": partition_density(G, partition),
        "num_clusters": partition_num_clusters(partition),
        "max_cluster_size": max(sizes),
        "avg_cluster_size": sum(sizes) / len(sizes),
        "runtime": runtime_seconds,
    }


def run_dataset(
        data_dir: Path,
        algorithms: list[AlgorithmSpec],
        dataset_name: str,
        size_class: str,
        graph_type: str,
        regime: str
) -> list[dict[str, object]]:
    """
   Runs multiple algorithms on all graph instances contained in a dataset.
   One result row is generated for every combination of graph instance and algorithm.

   Args:
       data_dir (Path): Directory containing graph instances.
       algorithms (list[AlgorithmSpec]): Algorithms to evaluate.
       dataset_name (str): Dataset identifier used in result tables.
       size_class (str): Graph size category.
       graph_type (str): Graph generator type.
       regime (str): Density regime.

   Returns:
       list[dict[str, object]]: A list of result rows that can be converted into a DataFrame.
   """
    rows = []

    for instance in load_instances_json(data_dir):
        G = instance.graph

        for algorithm in algorithms:
            result = run_algorithm(G, algorithm)

            rows.append(
                {
                    "dataset": dataset_name,
                    "size_class": size_class,
                    "graph_type": graph_type,
                    "regime": regime,
                    "instance": instance.name,
                    "n": G.number_of_nodes(),
                    "m": G.number_of_edges(),
                    "edge_density": edge_density(G),
                    **result,
                }
            )
    return rows


def add_relative_scores(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    Adds relative performance scores to a result table.

    Args:
        raw_results (pd.DataFrame): Per-instance experimental results.

    Returns:
        pd.DataFrame: Results including ``relative_to_best`` and ``is_best`` columns.
    """
    results = raw_results.copy()

    best_by_instance = results.groupby("instance")["density"].transform("max")
    results["relative_to_best"] = results["density"] / best_by_instance
    results["is_best"] = results["density"] == best_by_instance

    return results


def add_instance_ranks(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    Algorithms are ranked by decreasing density for every graph instance. Lower ranks indicate better solutions.

    Args:
        raw_results (pd.DataFrame): Per-instance experimental results.

    Returns:
        pd.DataFrame: Results including a ``rank`` column.
    """
    results = raw_results.copy()

    results["rank"] = results.groupby("instance")["density"].rank(method="min", ascending=False)

    return results


def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates experimental results over all instances of a dataset category.

    Args:
        raw_results (pd.DataFrame): Per-instance experimental results.

    Returns:
        pd.DataFrame: Aggregated summary table.
    """
    results = add_relative_scores(raw_results)
    results = add_instance_ranks(results)

    summary = (
        results.groupby(["size_class", "graph_type", "regime", "algorithm"])
        .agg(
            instances=("instance", "count"),
            mean_density=("density", "mean"),
            mean_relative_to_best=("relative_to_best", "mean"),
            mean_rank=("rank", "mean"),
            wins=("is_best", "sum"),
            mean_runtime_seconds=("runtime", "mean"),
            median_runtime_seconds=("runtime", "median"),
            mean_num_clusters=("num_clusters", "mean"),
            mean_max_cluster_size=("max_cluster_size", "mean"),
            mean_avg_cluster_size=("avg_cluster_size", "mean"),
        )
        .reset_index()
    )

    return summary


def thesis_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a compact summary table suitable for inclusion in reports.
    Only the most relevant evaluation metrics are retained and the table is sorted by graph category and average rank.

    Args:
        summary (pd.DataFrame): Aggregated summary statistics.

    Returns:
        pd.DataFrame: Compact summary table.
    """
    table = summary[
        [
            "size_class",
            "graph_type",
            "regime",
            "algorithm",
            "mean_relative_to_best",
            "mean_rank",
            "wins",
            "mean_runtime_seconds",
            "mean_num_clusters",
        ]
    ].copy()

    table = table.sort_values(
        ["size_class", "graph_type", "regime", "mean_rank"]
    )

    return table


def overall_thesis_summary_table(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    Computes an overall ranking across all datasets and graph categories.

    Results are aggregated by algorithm only and provide a global comparison of solution quality, runtime, and clustering behavior.

    Args:
        raw_results (pd.DataFrame): Per-instance experimental results.

    Returns:
        pd.DataFrame: Overall algorithm comparison table.
    """
    results = add_relative_scores(raw_results)
    results = add_instance_ranks(results)

    table = (
        results.groupby("algorithm")
        .agg(
            mean_relative_to_best=("relative_to_best", "mean"),
            mean_rank=("rank", "mean"),
            wins=("is_best", "sum"),
            mean_runtime_seconds=("runtime", "mean"),
            mean_num_clusters=("num_clusters", "mean"),
        )
        .reset_index()
        .sort_values("mean_rank")
    )

    return table


def rounded_for_export(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """
    Rounds all floating-point columns of a DataFrame for export.

    Args:
        df (pd.DataFrame): Input table.
        decimals (int): Number of decimal places. Defaults to 4.

    Returns:
        pd.DataFrame: Rounded copy of the input table.
    """
    result = df.copy()
    float_columns = result.select_dtypes(include="float").columns
    result[float_columns] = result[float_columns].round(decimals)
    return result