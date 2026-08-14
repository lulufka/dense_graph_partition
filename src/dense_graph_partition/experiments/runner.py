import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx
import pandas as pd

from dense_graph_partition.core.evaluation import edge_density, partition_cluster_sizes, partition_density, \
    partition_num_clusters, validate_partition
from dense_graph_partition.core.graph_io import load_instances_json, partition_to_json
from dense_graph_partition.core.types import Partition
from dense_graph_partition.experiments.datasets import build_datasets
from dense_graph_partition.experiments.run_tasks import run_tasks


@dataclass(frozen=True)
class GraphTask:
    """
    Stores graph and dataset information shared by experiment tasks.

    Attributes:
        dataset (str): Name of the dataset.
        size_class (str): Graph size category.
        graph_type (str): Graph generator type.
        regime (str): Density regime.
        instance_name (str): Name of the graph instance.
        graph (nx.Graph): Input graph.
    """
    dataset: str
    size_class: str
    graph_type: str
    regime: str
    instance_name: str
    graph: nx.Graph


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
class AlgorithmTask(GraphTask):
    """
    Represents one algorithm evaluation task.

    Attributes:
        algorithms (tuple[AlgorithmSpec, ...]): Algorithms evaluated on the instance.
    """
    algorithms: tuple[AlgorithmSpec, ...]


def graph_metadata(task: GraphTask) -> dict[str, object]:
    """
    Creates common graph and dataset metadata for an experiment task.

    Args:
        task (GraphTask): Task containing graph and dataset information.

    Returns:
        dict[str, object]: Graph and dataset metadata.
    """
    graph = task.graph

    return {
        "dataset": task.dataset,
        "size_class": task.size_class,
        "graph_type": task.graph_type,
        "regime": task.regime,
        "instance": task.instance_name,
        "n": graph.number_of_nodes(),
        "m": graph.number_of_edges(),
        "edge_density": edge_density(graph),
    }


def partition_stats(partition: Partition) -> dict[str, float | int]:
    """
    Computes basic partition statistics.

    Args:
        partition (Partition): Partition to evaluate.

    Returns:
        dict[str, float | int]: Number of clusters, maximum cluster size, and average cluster size.
    """
    sizes = partition_cluster_sizes(partition)

    return {
        "num_clusters": partition_num_clusters(partition),
        "max_cluster_size": max(sizes),
        "avg_cluster_size": sum(sizes) / len(sizes),
    }


def run_algorithm(G: nx.Graph, algorithm: AlgorithmSpec, include_partition: bool = False) -> dict[str, object]:
    """
    Executes one partitioning algorithm on a graph and computes evaluation statistics.

    Args:
        G (nx.Graph): Input graph.
        algorithm (AlgorithmSpec): Algorithm to execute.

    Returns:
        dict[str, object]: Evaluation metrics and runtime information.
    """
    start_time = time.perf_counter()
    partition = algorithm.run(G)
    runtime = time.perf_counter() - start_time

    validate_partition(G, partition)

    result = {
        "algorithm": algorithm.name,
        "density": partition_density(G, partition),
        **partition_stats(partition),
        "runtime": runtime,
    }

    if include_partition:
        result["partition"] = partition_to_json(partition)

    return result


def evaluate_algorithm_task(task: AlgorithmTask) -> list[dict[str, object]]:
    """
    Evaluates all specified algorithms on one graph instance.

    Args:
        task (AlgorithmTask): Evaluation task containing graph, dataset metadata, and algorithms.

    Returns:
        list[dict[str, object]]: Result rows for all evaluated algorithms.
    """
    metadata = graph_metadata(task)

    rows: list[dict[str, object]] = []

    for algorithm in task.algorithms:
        result = run_algorithm(task.graph, algorithm)
        rows.append({**metadata, **result})

    return rows


def build_algorithm_tasks(data_root: Path, algorithms: list[AlgorithmSpec]) -> list[AlgorithmTask]:
    """
    Builds algorithm evaluation tasks for all graph instances.

    Args:
        data_root (Path): Root directory containing generated graph instances.
        algorithms (list[AlgorithmSpec]): Algorithms to evaluate.

    Returns:
        list[AlgorithmTask]: Algorithm evaluation tasks.
    """
    tasks: list[AlgorithmTask] = []

    algorithm_tuple = tuple(algorithms)

    for dataset in build_datasets(data_root):
        if not dataset.path.exists():
            raise FileNotFoundError(f"Dataset directory {dataset.path} does not exist.")

        instances = load_instances_json(dataset.path)

        for instance in instances:
            tasks.append(
                AlgorithmTask(
                    dataset=dataset.name,
                    size_class=dataset.size_class,
                    graph_type=dataset.graph_type,
                    regime=dataset.regime,
                    instance_name=instance.name,
                    graph=instance.graph,
                    algorithms=algorithm_tuple,
                )
            )

    return tasks


def run_algorithm_tasks(tasks: list[AlgorithmTask], workers: int) -> list[dict[str, object]]:
    """
    Runs algorithm evaluation tasks sequentially or in parallel.

    Args:
        tasks (list[AlgorithmTask]): Tasks to evaluate.
        workers (int): Number of worker processes.

    Returns:
        list[dict[str, object]]: Raw result rows.
    """
    task_results = run_tasks(
        tasks=tasks,
        evaluate=evaluate_algorithm_task,
        workers=workers,
        describe=lambda task: f"{task.dataset} | {task.instance_name}",
    )

    return [row for task_rows in task_results for row in task_rows]


def write_raw_results(raw_results: pd.DataFrame, results_dir: Path) -> None:
    """
    Writes raw algorithm evaluation results to a CSV file.

    Args:
        raw_results (pd.DataFrame): Raw algorithm result table.
        results_dir (Path): Output directory.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_results.to_csv(results_dir / "raw_results.csv", index=False)
