from dataclasses import dataclass
from pathlib import Path

from dense_graph_partition.core.graph_io import load_ground_truth_graph_json
from dense_graph_partition.core.types import Partition
from dense_graph_partition.experiments.run_tasks import run_tasks
from dense_graph_partition.experiments.runner import AlgorithmSpec, graph_metadata, AlgorithmTask, run_algorithm


@dataclass(frozen=True)
class GroundTruthTask(AlgorithmTask):
    mu: float
    ground_truth: Partition


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
        "mu": task.mu,
        "ground_truth_num_clusters": len(task.ground_truth),
    }

    rows: list[dict[str, object]] = []

    for algorithm in task.algorithms:
        result = run_algorithm(G=task.graph, algorithm=algorithm, include_partition=True)
        rows.append({**metadata, **result,})

    return rows


def build_ground_truth_tasks(data_root: Path, algorithms: list[AlgorithmSpec]) -> list[GroundTruthTask]:
    """
    Builds ground-truth evaluation tasks for all random partition graph instances.

    Args:
        data_root (Path): Root directory containing ground-truth graph instances.
        algorithms (list[AlgorithmSpec]): Algorithms to evaluate.

    Returns:
        list[GroundTruthTask]: Ground-truth evaluation tasks.
    """
    tasks: list[GroundTruthTask] = []

    algorithm_tuple = tuple(algorithms)

    if not data_root.exists():
        raise FileNotFoundError(f"Ground-truth data directory does not exist: {data_root}")

    for size_class_dir in sorted(data_root.iterdir()):
        if not size_class_dir.is_dir():
            continue

        size_class = size_class_dir.name

        for graph_type_dir in sorted(size_class_dir.iterdir()):
            if not graph_type_dir.is_dir():
                continue

            graph_type = graph_type_dir.name

            for regime_dir in sorted(graph_type_dir.iterdir()):
                if not regime_dir.is_dir():
                    continue

                regime = regime_dir.name

                for mu_dir in sorted(regime_dir.iterdir()):
                    if not mu_dir.is_dir():
                        continue

                    for instance_path in sorted(mu_dir.glob("*.json")):
                        instance = load_ground_truth_graph_json(instance_path)

                        tasks.append(
                            GroundTruthTask(
                                dataset=f"{size_class}_{graph_type}_{regime}_mu_{instance.mu}",
                                size_class=size_class,
                                graph_type=graph_type,
                                regime=regime,
                                instance_name=instance.name,
                                graph=instance.graph,
                                mu=instance.mu,
                                ground_truth=instance.ground_truth,
                                algorithms=algorithm_tuple,
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
        describe=lambda task: f"{task.size_class} | {task.graph_type} | {task.regime} | mu={task.mu} | {task.instance_name}",
    )

    return [row for task_rows in task_results for row in task_rows]
