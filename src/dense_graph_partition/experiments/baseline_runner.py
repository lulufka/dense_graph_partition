import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

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


@dataclass(frozen=True)
class BaselineTask:
    """
    Represents one baseline evaluation task.
    One task evaluates all baseline algorithms on a single graph instance.

    Attributes:
        dataset_name (str): Name of the dataset.
        size_class (str): Graph size category.
        graph_type (str): Graph generator type.
        regime (str): Density regime.
        instance_name (str): Name of the graph instance.
        graph (nx.Graph): Input graph.
    """

    dataset_name: str
    size_class: str
    graph_type: str
    regime: str
    instance_name: str
    graph: nx.Graph


def evaluate_baseline_task(task: BaselineTask) -> list[dict[str, object]]:
    """
    Evaluates all baseline algorithms on one graph instance.
    The graph is processed once and all baseline algorithms are executed sequentially. One result row is produced for every algorithm.

    Args:
        task (BaselineTask): Evaluation task containing graph and dataset metadata.

    Returns:
        list[dict[str, object]]: Result rows for all evaluated algorithms.
    """
    from dense_graph_partition.experiments.algorithm_registry import build_baseline_algorithm_specs

    graph = task.graph
    algorithms = build_baseline_algorithm_specs()

    rows: list[dict[str, object]] = []

    graph_metadata = {
        "dataset": task.dataset_name,
        "size_class": task.size_class,
        "graph_type": task.graph_type,
        "regime": task.regime,
        "instance": task.instance_name,
        "n": graph.number_of_nodes(),
        "m": graph.number_of_edges(),
        "edge_density": edge_density(graph),
    }

    for algorithm in algorithms:
        result = run_algorithm(graph, algorithm)

        rows.append({**graph_metadata, **result})

    return rows


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


def run_dataset(data_dir: Path, dataset_name: str, size_class: str, graph_type: str, regime: str, workers: int = 1) -> list[dict[str, object]]:
    """
    Runs all baseline algorithms on all graph instances of one dataset.
    Each graph instance forms one parallel task. All algorithms are evaluated sequentially inside that task.

   Args:
       data_dir (Path): Directory containing graph instances.
       dataset_name (str): Dataset identifier used in result tables.
       size_class (str): Graph size category.
       graph_type (str): Graph generator type.
       regime (str): Density regime.
       workers (int): Number of worker processes. A value of 1 disables parallelization.

    Returns:
        list[dict[str, object]]: One result row for every graph-instance/algorithm combination.
    """
    instances = load_instances_json(data_dir)

    tasks = [
        BaselineTask(dataset_name, size_class, graph_type, regime, instance.name, instance.graph)
        for instance in instances
    ]

    rows: list[dict[str, object]] = []
    total_tasks = len(tasks)

    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            task_rows = evaluate_baseline_task(task)
            rows.extend(task_rows)

            print(f"[{index}/{total_tasks}]  {dataset_name} | {task.instance_name}", flush=True)

        return rows

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_baseline_task, task): task for task in tasks}

        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]

            try:
                task_rows = future.result()
            except Exception as error:
                raise RuntimeError(f"Baseline task failed: dataset={task.dataset_name}, instance={task.instance_name}") from error

            rows.extend(task_rows)

            print(f"[{index}/{total_tasks}] {dataset_name} | {task.instance_name}", flush=True)

    return rows


def add_relative_scores(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    Adds relative performance scores to a result table.
    The relative score is the quotient between the best density found for an instance and the density achieved by the respective algorithm.
    A value of 1.0 indicates the best solution, while larger values indicate lower solution quality.

    Args:
        raw_results (pd.DataFrame): Per-instance experimental results.

    Returns:
        pd.DataFrame: Results including ``relative_to_best`` and ``is_best`` columns.
    """
    results = raw_results.copy()

    best_by_instance = results.groupby(["dataset", "instance"])["density"].transform("max")
    results["relative_to_best"] = best_by_instance / results["density"]
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

    results["rank"] = results.groupby(["dataset", "instance"])["density"].rank(method="min", ascending=False)

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