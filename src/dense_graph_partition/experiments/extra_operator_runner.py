import time
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
from dense_graph_partition.local_search.pipeline import PipelineStepResult, offset_step_results, run_local_search_pipeline


@dataclass(frozen=True)
class BranchingLocalSearchExperiment:
    """
    Describes a local-search experiment with a shared prefix and multiple branches.

    Attributes:
        name (str): Human-readable experiment name.
        start_partition (str): Start-partition algorithm.
        shared_prefix (str): Pipeline computed once before branching.
        suffixes (tuple[str, ...]): Pipelines applied independently to copies of the shared-prefix result.
        zero_gain_factor (int): Multiplier used to determine the maximum number of consecutive zero-gain moves.
    """
    name: str
    start_partition: str
    shared_prefix: str
    suffixes: tuple[str, ...]
    zero_gain_factor: int = 4


@dataclass(frozen=True)
class BranchingLocalSearchTask(GraphTask):
    """
    Stores one branching local-search evaluation task.

    Attributes:
        run (int): Run number.
        seed (int): Random seed used for randomized components.
        experiment (BranchingLocalSearchExperiment): Local-search configuration.
    """
    run: int
    seed: int
    experiment: BranchingLocalSearchExperiment



def build_extra_operator_experiments(start_partitions: list[str]) -> list[BranchingLocalSearchExperiment]:
    experiments: list[BranchingLocalSearchExperiment] = []

    shared_prefix = ",".join(["move_plateau"] * 4)

    suffixes = (
        "",
        "merge_first",
        "merge_best",
        "bridge_split",
        "split_min_cut",
    )

    for start_partition in start_partitions:
        experiments.append(
            BranchingLocalSearchExperiment(
                name=f"{start_partition}_extra_operators",
                start_partition=start_partition,
                shared_prefix=shared_prefix,
                suffixes=suffixes,
            )
        )

    return experiments


def copy_partition(partition: Partition) -> Partition:
    """
    Creates an independent copy of a partition.

    Args:
        partition (Partition): Partition to copy.

    Returns:
        Partition: Copy whose clusters can be modified independently.
    """
    return [set(cluster) for cluster in partition]


def build_raw_result_row(
        task: BranchingLocalSearchTask,
        pipeline: str,
        extra_operator: str,
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
        "experiment": task.experiment.name,
        "start_partition": task.experiment.start_partition,
        "pipeline": pipeline,
        "extra_operator": extra_operator,
        **graph_metadata(task),
        "run": task.run,
        "zero_gain_factor": task.experiment.zero_gain_factor,
        "seed": task.seed,
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


def step_rows_from_pipeline_steps(task: BranchingLocalSearchTask, pipeline: str, extra_operator: str, steps: list[PipelineStepResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    metadata = graph_metadata(task)

    for step in steps:
        rows.append(
            {
                "experiment": task.experiment.name,
                "start_partition": task.experiment.start_partition,
                "pipeline": pipeline,
                "extra_operator": extra_operator,
                **metadata,
                "run": task.run,
                "zero_gain_factor": task.experiment.zero_gain_factor,
                "seed": task.seed,
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

def evaluate_branching_local_search_task(task: BranchingLocalSearchTask) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    experiment = task.experiment

    graph = task.graph
    start_algorithm = build_partition_algorithm(experiment.start_partition)

    start_time = time.perf_counter()
    start_partition = start_algorithm(graph)
    start_runtime = time.perf_counter() - start_time

    prefix_start = time.perf_counter()
    prefix_result = run_local_search_pipeline(
        G=graph,
        partition=copy_partition(start_partition),
        pipeline=experiment.shared_prefix,
        zero_gain_factor=experiment.zero_gain_factor,
        random_seed=task.seed,
    )
    prefix_runtime = time.perf_counter() - prefix_start

    shared_partition = prefix_result.partition

    raw_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for suffix_index, suffix in enumerate(experiment.suffixes):
        if suffix:
            extra_operator = suffix
            suffix_seed = task.seed + 10_000 + 1_000 * suffix_index

            suffix_start = time.perf_counter()
            suffix_result = run_local_search_pipeline(
                G=graph,
                partition=copy_partition(shared_partition),
                pipeline=suffix,
                zero_gain_factor=experiment.zero_gain_factor,
                random_seed=suffix_seed,
            )
            suffix_runtime = time.perf_counter() - suffix_start

            full_pipeline = f"{experiment.shared_prefix},{suffix}"
            final_partition = suffix_result.partition

            num_moves = prefix_result.num_moves + suffix_result.num_moves
            num_passes = prefix_result.num_passes + suffix_result.num_passes

            full_steps = list(prefix_result.steps) + offset_step_results(suffix_result.steps, offset=len(prefix_result.steps))

            effective_ls_runtime = prefix_runtime + suffix_runtime

        else:
            extra_operator = "baseline"
            full_pipeline = experiment.shared_prefix
            final_partition = copy_partition(shared_partition)

            num_moves = prefix_result.num_moves
            num_passes = prefix_result.num_passes
            full_steps = list(prefix_result.steps)
            effective_ls_runtime = prefix_runtime

        raw_rows.append(
            build_raw_result_row(
                task=task,
                pipeline=full_pipeline,
                extra_operator=extra_operator,
                start_partition=start_partition,
                final_partition=final_partition,
                start_runtime=start_runtime,
                ls_runtime=effective_ls_runtime,
                num_moves=num_moves,
                num_passes=num_passes,
            )
        )

        step_rows.extend(
            step_rows_from_pipeline_steps(
                task=task,
                pipeline=full_pipeline,
                extra_operator=extra_operator,
                steps=full_steps,
            )
        )

    return raw_rows, step_rows


def build_branching_local_search_tasks(data_root: Path, experiments: list[BranchingLocalSearchExperiment], runs: int, base_seed: int) -> list[BranchingLocalSearchTask]:
    """
    Builds all branching local-search evaluation tasks.

    Args:
        data_root (Path): Root directory containing generated graph instances.
        experiments (list[BranchingLocalSearchExperiment]): Experiments to run.
        runs (int): Number of runs per instance and experiment.
        base_seed (int): Base seed for reproducibility.

    Returns:
        list[BranchingLocalSearchTask]: Evaluation tasks.
    """
    tasks: list[BranchingLocalSearchTask] = []

    global_instance_index = 0

    for dataset in build_datasets(data_root):
        if not dataset.path.exists():
            raise FileNotFoundError(f"Dataset directory {dataset.path} does not exist.")

        instances = load_instances_json(dataset.path)

        for instance in instances:
            for run_index in range(runs):
                seed = (
                        base_seed
                        + 1_000_000 * run_index
                        + 1_000 * global_instance_index
                )

                for experiment in experiments:
                    tasks.append(
                        BranchingLocalSearchTask(
                            dataset=dataset.name,
                            size_class=dataset.size_class,
                            graph_type=dataset.graph_type,
                            regime=dataset.regime,
                            instance_name=instance.name,
                            graph=instance.graph,
                            run=run_index,
                            seed=seed,
                            experiment=experiment,
                        )
                    )

            global_instance_index += 1

    return tasks


def run_branching_local_search_tasks(tasks: list[BranchingLocalSearchTask], workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Runs local-search tasks sequentially or in parallel.

    Args:
        tasks (list[BranchingLocalSearchTask]): Tasks to evaluate.
        workers (int): Number of worker processes.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]: Raw pipeline rows and step-level rows.
    """
    task_results = run_tasks(
        tasks=tasks,
        evaluate=evaluate_branching_local_search_task,
        workers=workers,
        describe=lambda task: (f"{task.experiment.name} | {task.instance_name}"),
    )

    raw_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for task_raw_rows, task_step_rows in task_results:
        raw_rows.extend(task_raw_rows)
        step_rows.extend(task_step_rows)

    return raw_rows, step_rows


def write_branching_local_search_results(raw_results: pd.DataFrame, step_results: pd.DataFrame, results_dir: Path) -> None:
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
