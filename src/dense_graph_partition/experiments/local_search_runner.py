import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dense_graph_partition.core.types import Partition
from dense_graph_partition.core.evaluation import partition_density
from dense_graph_partition.core.graph_io import load_instances_json
from dense_graph_partition.experiments.algorithm_registry import build_partition_algorithm
from dense_graph_partition.experiments.datasets import build_datasets
from dense_graph_partition.experiments.run_tasks import run_tasks
from dense_graph_partition.experiments.runner import GraphTask, partition_stats, graph_metadata
from dense_graph_partition.local_search.pipeline import PipelineStepResult, run_local_search_pipeline


@dataclass(frozen=True)
class LocalSearchExperiment:
    """
    Describes one local-search experiment configuration.

    Attributes:
        name (str): Human-readable experiment name.
        start_partition (str): Name of the start partition algorithm.
        pipeline (str): Comma-separated local-search pipeline.
        zero_gain_factor (int): Multiplier used to determine the maximum number of consecutive zero-gain moves.
    """
    name: str
    start_partition: str
    pipeline: str
    zero_gain_factor: int | None = None


@dataclass(frozen=True)
class LocalSearchTask(GraphTask):
    """
    Stores all local-search evaluations for one graph instance and one start-partition algorithm.

    Attributes:
        start_partition_name (str): Start-partition algorithm evaluated in this task.
        experiments (tuple[LocalSearchExperiment, ...]): Local-search configurations.
        runs (int): Number of randomized runs.
        base_seed (int): Base seed for reproducibility.
        instance_index (int): Globally unique instance index used for seed generation.
    """
    start_partition_name: str
    experiments: tuple[LocalSearchExperiment, ...]
    runs: int
    base_seed: int
    instance_index: int


def group_experiments_by_start_partition(experiments: list[LocalSearchExperiment]) -> dict[str, tuple[LocalSearchExperiment, ...]]:
    grouped: dict[str, list[LocalSearchExperiment]] = defaultdict(list)

    for experiment in experiments:
        grouped[experiment.start_partition].append(experiment)

    return {start_partition: tuple(group) for start_partition, group in grouped.items()}


def build_local_search_experiments(start_partitions: list[str]) -> list[LocalSearchExperiment]:
    """
    Builds all local-search experiment configurations.

    Args:
        start_partitions (list[str]): Names of the start-partition algorithms.

    Returns:
        list[LocalSearchExperiment]: Concrete local-search experiment configurations.
    """
    experiments: list[LocalSearchExperiment] = []

    plateau_pipeline = ",".join(["move_plateau"] * 10)

    for start_partition in start_partitions:
        # experiments.append(
        #     LocalSearchExperiment(
        #         name=f"{start_partition}_move_first",
        #         start_partition=start_partition,
        #         pipeline="move_first",
        #     )
        # )
#
        # experiments.append(
        #     LocalSearchExperiment(
        #         name=f"{start_partition}_move_best",
        #         start_partition=start_partition,
        #         pipeline="move_best",
        #     )
        # )

        for factor in (2,):
            experiments.append(
                LocalSearchExperiment(
                    name=(f"{start_partition}_move_plateau10_zg{factor}"),
                    start_partition=start_partition,
                    pipeline=plateau_pipeline,
                    zero_gain_factor=factor,
                )
            )

    return experiments


def copy_partition(partition: Partition) -> Partition:
    """
    Creates a deep copy of a partition.

    Args:
        partition (Partition): Partition to copy.

    Returns:
        Partition: Independent copy.
    """
    return [set(cluster) for cluster in partition]


def build_raw_result_row(
        task: LocalSearchTask,
        experiment: LocalSearchExperiment,
        run: int,
        seed: int,
        pipeline: str,
        start_partition: Partition,
        final_partition: Partition,
        start_runtime: float,
        ls_runtime: float,
        num_moves: int,
        num_passes: int,
) -> dict[str, Any]:
    graph = task.graph

    start_density = partition_density(graph, start_partition)
    final_density = partition_density(graph, final_partition)

    start_stats = partition_stats(start_partition)
    final_stats = partition_stats(final_partition)

    return {
        "experiment": experiment.name,
        "start_partition": experiment.start_partition,
        "pipeline": pipeline,
        **graph_metadata(task),
        "run": run,
        "zero_gain_factor": experiment.zero_gain_factor,
        "seed": seed,
        "start_density": start_density,
        "final_density": final_density,
        "improved": final_density > start_density,
        "num_moves": num_moves,
        "num_passes": num_passes,
        "start_runtime": start_runtime,
        "ls_runtime": ls_runtime,
        "total_runtime": start_runtime + ls_runtime,
        "start_num_clusters": start_stats["num_clusters"],
        "final_num_clusters": final_stats["num_clusters"],
        "start_max_cluster_size": start_stats["max_cluster_size"],
        "final_max_cluster_size": final_stats["max_cluster_size"],
        "start_avg_cluster_size": start_stats["avg_cluster_size"],
        "final_avg_cluster_size": final_stats["avg_cluster_size"],
    }


def step_rows_from_pipeline_steps(
        task: LocalSearchTask,
        experiment: LocalSearchExperiment,
        run: int,
        seed: int,
        pipeline: str,
        steps: list[PipelineStepResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    metadata = graph_metadata(task)

    for step in steps:
        rows.append(
            {
                "experiment": experiment.name,
                "start_partition": experiment.start_partition,
                "pipeline": pipeline,
                **metadata,
                "run": run,
                "zero_gain_factor": experiment.zero_gain_factor,
                "seed": seed,
                "step_index": step.step_index,
                "step_name": step.step_name,
                "score_before": step.score_before,
                "score_after": step.score_after,
                "num_moves": step.num_moves,
                "num_passes": step.num_passes,
                "runtime": step.runtime,
                "num_clusters_before": step.num_clusters_before,
                "num_clusters_after": step.num_clusters_after,
            }
        )

    return rows


def evaluate_local_search_task(task: LocalSearchTask) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Evaluates one local-search task.

    Args:
        task (LocalSearchTask): Task containing graph, seed, start partition, and pipeline.

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]]]: One raw pipeline result row and one list of step-level rows.
    """
    graph = task.graph

    start_algorithm = build_partition_algorithm(task.start_partition_name)

    start_time = time.perf_counter()
    start_partition = start_algorithm(graph)
    start_runtime = time.perf_counter() - start_time

    raw_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for run_index in range(task.runs):
        seed = (
                task.base_seed
                + 1_000_000 * run_index
                + 1_000 * task.instance_index
        )

        for experiment in task.experiments:
            zero_gain_factor = (
                4
                if experiment.zero_gain_factor is None
                else experiment.zero_gain_factor
            )

            ls_start = time.perf_counter()

            ls_result = run_local_search_pipeline(
                G=graph,
                partition=copy_partition(start_partition),
                pipeline=experiment.pipeline,
                zero_gain_factor=zero_gain_factor,
                random_seed=seed,
            )

            ls_runtime = time.perf_counter() - ls_start

            raw_rows.append(
                build_raw_result_row(
                    task=task,
                    experiment=experiment,
                    run=run_index,
                    seed=seed,
                    pipeline=experiment.pipeline,
                    start_partition=start_partition,
                    final_partition=ls_result.partition,
                    start_runtime=start_runtime,
                    ls_runtime=ls_runtime,
                    num_moves=ls_result.num_moves,
                    num_passes=ls_result.num_passes,
                )
            )

            step_rows.extend(
                step_rows_from_pipeline_steps(
                    task=task,
                    experiment=experiment,
                    run=run_index,
                    seed=seed,
                    pipeline=experiment.pipeline,
                    steps=ls_result.steps,
                )
            )

    return raw_rows, step_rows


def build_local_search_tasks(data_root: Path, experiments: list[LocalSearchExperiment], runs: int, base_seed: int) -> list[LocalSearchTask]:
    """
    Builds all local-search evaluation tasks.

    Args:
        data_root (Path): Root directory containing generated graph instances.
        experiments (list[LocalSearchExperiment]): Experiments to run.
        runs (int): Number of runs per instance and experiment.
        base_seed (int): Base seed for reproducibility.

    Returns:
        list[LocalSearchTask]: Evaluation tasks.
    """
    tasks: list[LocalSearchTask] = []

    experiments_by_start_partition = group_experiments_by_start_partition(experiments)

    global_instance_index = 0

    for dataset in build_datasets(data_root):
        if not dataset.path.exists():
            raise FileNotFoundError(f"Dataset directory {dataset} does not exist.")

        instances = load_instances_json(dataset.path)

        for instance in instances:
            for start_partition_name, grouped_experiments in experiments_by_start_partition.items():
                tasks.append(
                    LocalSearchTask(
                        dataset=dataset.name,
                        size_class=dataset.size_class,
                        graph_type=dataset.graph_type,
                        regime=dataset.regime,
                        instance_name=instance.name,
                        graph=instance.graph,
                        start_partition_name=start_partition_name,
                        experiments=grouped_experiments,
                        runs=runs,
                        base_seed=base_seed,
                        instance_index=global_instance_index,
                    )
                )

            global_instance_index += 1

    return tasks


def run_local_search_tasks(tasks: list[LocalSearchTask], workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Runs local-search tasks sequentially or in parallel.

    Args:
        tasks (list[LocalSearchTask]): Tasks to evaluate.
        workers (int): Number of worker processes.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]: Raw pipeline rows and step-level rows.
    """
    task_results = run_tasks(
        tasks=tasks,
        evaluate=evaluate_local_search_task,
        workers=workers,
        describe=lambda task: (f"{task.start_partition_name} | {task.instance_name}"),
    )

    raw_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for task_raw_rows, task_step_rows in task_results:
        raw_rows.extend(task_raw_rows)
        step_rows.extend(task_step_rows)

    return raw_rows, step_rows


def write_local_search_results(raw_results: pd.DataFrame, step_results: pd.DataFrame, results_dir: Path) -> None:
    """
    Writes local-search results to CSV files.

    Args:
        raw_results (pd.DataFrame): Raw pipeline result table.
        step_results (pd.DataFrame): Step-level result table.
        results_dir (Path): Output directory.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_results.to_csv(results_dir / "raw_results.csv", index=False)
    step_results.to_csv(results_dir / "step_results.csv", index=False)
