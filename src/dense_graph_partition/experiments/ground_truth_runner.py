from dataclasses import dataclass
from pathlib import Path

from dense_graph_partition.core.evaluation import partition_density
from dense_graph_partition.core.graph_io import load_ground_truth_graph_json, partition_to_json
from dense_graph_partition.core.types import Partition
from dense_graph_partition.experiments.run_tasks import run_tasks
from dense_graph_partition.experiments.runner import AlgorithmSpec, AlgorithmTask, graph_metadata, partition_stats, run_algorithm


@dataclass(frozen=True)
class GroundTruthTask(AlgorithmTask):
    """
    Represents one ground-truth evaluation task.

    Attributes:
        ground_truth (Partition): Ground-truth partition of the graph.
        community_size_class (str): Community size regime.
        generation_metadata (dict[str, object]): Metadata stored during graph generation.
    """
    ground_truth: Partition
    community_size_class: str
    generation_metadata: dict[str, object]


def ground_truth_metadata(task: GroundTruthTask) -> dict[str, object]:
    """
    Computes metadata and statistics of the ground-truth partition.

    Args:
        task (GroundTruthTask): Ground-truth evaluation task.

    Returns:
        dict[str, object]: Ground-truth metadata and statistics.
    """
    stats = partition_stats(task.ground_truth)

    return {
        "community_size_class": task.community_size_class,
        "ground_truth_density": partition_density(task.graph, task.ground_truth),
        "ground_truth_num_clusters": stats["num_clusters"],
        "ground_truth_max_cluster_size": stats["max_cluster_size"],
        "ground_truth_avg_cluster_size": stats["avg_cluster_size"],
        "ground_truth_partition": partition_to_json(task.ground_truth),
        **task.generation_metadata,
    }


def evaluate_ground_truth_task(task: GroundTruthTask) -> list[dict[str, object]]:
    """
    Evaluates all specified algorithms on one graph instance with ground truth.

    Args:
        task (GroundTruthTask): Ground-truth evaluation task.

    Returns:
        list[dict[str, object]]: Result rows for all evaluated algorithms.
    """
    metadata = {
        **graph_metadata(task),
        **ground_truth_metadata(task),
    }

    rows: list[dict[str, object]] = []

    for algorithm in task.algorithms:
        result = run_algorithm(G=task.graph, algorithm=algorithm, include_partition=True)
        rows.append({**metadata, **result,})

    return rows


def build_ground_truth_tasks(data_root: Path, algorithms: list[AlgorithmSpec]) -> list[GroundTruthTask]:
    """
    Builds ground-truth evaluation tasks for all JSON instances below the given data root.

    Args:
        data_root (Path): Root directory containing ground-truth graph instances.
        algorithms (list[AlgorithmSpec]): Algorithms to evaluate.

    Returns:
        list[GroundTruthTask]: Ground-truth evaluation tasks.
    """
    if not data_root.exists():
        raise FileNotFoundError(f"Ground-truth data directory does not exist: {data_root}")

    instance_paths = sorted(data_root.rglob("*.json"))

    if not instance_paths:
        raise FileNotFoundError(f"No ground-truth JSON instances found below {data_root}")

    algorithm_tuple = tuple(algorithms)

    tasks: list[GroundTruthTask] = []

    for instance_path in instance_paths:
        instance = load_ground_truth_graph_json(instance_path)

        metadata = instance.metadata

        try:
            size_class = str(metadata["size_class"])
            graph_type = str(metadata["graph_type"])
            regime = str(metadata["regime"])
            community_size_class = str(metadata["community_size_class"])
        except KeyError as exc:
            raise ValueError(f"Missing generation metadata {exc!s} in instance {instance_path}") from exc

        dataset = f"{size_class}_{graph_type}_{regime}_communities_{community_size_class}"


        tasks.append(
            GroundTruthTask(
                dataset=dataset,
                size_class=size_class,
                graph_type=graph_type,
                regime=regime,
                instance_name=instance.name,
                graph=instance.graph,
                algorithms=algorithm_tuple,
                ground_truth=instance.ground_truth,
                community_size_class=community_size_class,
                generation_metadata=metadata,
            )
        )

    return tasks


def run_ground_truth_tasks(tasks: list[GroundTruthTask], workers: int) -> list[dict[str, object]]:
    """
    Runs ground-truth evaluation tasks sequentially or in parallel.

    Args:
        tasks (list[GroundTruthTask]): Tasks to evaluate.
        workers (int): Number of worker processes.

    Returns:
        list[dict[str, object]]: Raw ground-truth result rows.
    """
    task_results = run_tasks(
        tasks=tasks,
        evaluate=evaluate_ground_truth_task,
        workers=workers,
        describe=lambda task: f"{task.size_class} | {task.graph_type} | {task.regime} | communities={task.community_size_class} | {task.instance_name}",
    )

    return [row for task_rows in task_results for row in task_rows]
