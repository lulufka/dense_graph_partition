import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from dense_graph_partition.core.evaluation import partition_density, partition_cluster_sizes, partition_num_clusters, \
    validate_partition, edge_density
from dense_graph_partition.core.graph_io import load_instances_json
from dense_graph_partition.experiments.algorithm_registry import build_partition_algorithm
from dense_graph_partition.experiments.baseline_runner import rounded_for_export
from dense_graph_partition.experiments.datasets import build_datasets
from dense_graph_partition.local_search.pipeline import PipelineResult, run_local_search_pipeline


@dataclass(frozen=True)
class LocalSearchExperiment:
    """
    Describes one local-search experiment configuration.

    Attributes:
        phase (str): Experiment phase. Used to separate output files.
        name (str): Human-readable experiment name.
        start_partition (str): Name of the start partition algorithm.
        pipeline (str): Comma-separated local-search pipeline.
    """
    phase: str
    name: str
    start_partition: str
    pipeline: str


@dataclass(frozen=True)
class LocalSearchTask:
    """
    Stores one evaluation task.

    Attributes:
        dataset (str): Dataset name.
        size_class (str): Size class of the dataset.
        graph_type (str): Type of generated graph.
        regime (str): Density regime.
        instance_name (str): Name of the graph instance.
        graph (nx.Graph): Input graph.
        run (int): One-based run number.
        seed (int): Random seed used for randomized components.
        experiment (LocalSearchExperiment): Local-search configuration.
    """
    dataset: str
    size_class: str
    graph_type: str
    regime: str
    instance_name: str
    graph: nx.Graph
    run: int
    seed: int
    experiment: LocalSearchExperiment


def build_local_search_experiments() -> list[LocalSearchExperiment]:
    """
    Builds all local-search pipeline configurations.

    Returns:
        list[LocalSearchExperiment]: Pipeline configurations without concrete start partitions.
    """
    single_pipelines = [
        "move_first",
        "move_best",
        "move_plateau",
        "merge_first",
        "merge_best",
        "split_min_cut",
        "bridge_split",
        "peel_node",
    ]

    short_pipelines = [
        "merge_best,move_best",
        "merge_first,move_best",
        "move_best,merge_best",
        "move_plateau,merge_best",
        "merge_best,move_plateau",
        "peel_node,move_best",
        "merge_best,peel_node,move_best",
        "merge_best,bridge_split,move_best",
        "merge_best,split_min_cut,move_best",
    ]

    long_pipelines = [
        "merge_best,move_best,peel_node,move_best",
        "merge_best,move_plateau,peel_node,move_best",
        "merge_first,move_plateau,merge_best,peel_node,move_best",
        "merge_best,move_best,bridge_split,move_best,peel_node,move_best",
        "merge_best,move_best,split_min_cut,move_best,peel_node,move_best",
        "move_plateau,merge_best,bridge_split,move_plateau,peel_node,move_best",
        "merge_best,move_best,peel_node,merge_best,bridge_split,move_best",
    ]

    experiments: list[LocalSearchExperiment] = []

    for index, pipeline in enumerate(single_pipelines):
        experiments.append(
            LocalSearchExperiment(
                phase="single",
                name=f"single_{index:02d}",
                start_partition="",
                pipeline=pipeline
            )
        )

    for index, pipeline in enumerate(short_pipelines):
        experiments.append(
            LocalSearchExperiment(
                phase="short",
                name=f"short_{index:02d}",
                start_partition="",
                pipeline=pipeline
            )
        )

    for index, pipeline in enumerate(long_pipelines):
        experiments.append(
            LocalSearchExperiment(
                phase="long",
                name=f"long_{index:02d}",
                start_partition="",
                pipeline=pipeline
            )
        )

    return experiments


def build_phase_experiments(phase: str, start_partitions: list[str]) -> list[LocalSearchExperiment]:
    """
    Builds experiments for the selected phase and all start partitions.

    Args:
        phase (str): Selected phase. Supported values are ``"single"``, `"short"``, ``"long"``, and ``"all"``.
        start_partitions (list[str]): Start partition names.

    Returns:
        list[LocalSearchExperiment]: Concrete experiment configurations.

    Raises:
        ValueError: If the selected phase is unknown.
    """
    if phase not in {"single", "short", "long", "all"}:
        raise ValueError(f"Unknown phase: {phase}")

    base_experiments = build_local_search_experiments()
    if phase != "all":
        base_experiments = [experiment for experiment in base_experiments if experiment.phase == phase]

    experiments: list[LocalSearchExperiment] = []

    for experiment in base_experiments:
        for start_partition in start_partitions:
            experiments.append(
                LocalSearchExperiment(
                    phase=experiment.phase,
                    name=f"{start_partition}_{experiment.name}",
                    start_partition=start_partition,
                    pipeline=experiment.pipeline
                )
            )

    return experiments


def partition_stats(partition: list[set[int]]) -> dict[str, float | int]:
    """
    Computes basic partition statistics.

    Args:
        partition (list[set[int]]): Partition to evaluate.

    Returns:
        dict[str, float | int]: Number of clusters, maximum cluster size,
        and average cluster size.
    """
    sizes = partition_cluster_sizes(partition)

    return {
        "num_clusters": partition_num_clusters(partition),
        "max_cluster_size": max(sizes),
        "avg_cluster_size": sum(sizes) / len(sizes),
    }


def step_rows_from_pipeline_result(task: LocalSearchTask, result: PipelineResult) -> list[dict[str, Any]]:
    """
    Converts pipeline step results into export rows.

    Args:
        task (LocalSearchTask): Evaluated task.
        result (PipelineResult): Local-search pipeline result.

    Returns:
        list[dict[str, Any]]: One row per pipeline step.
    """
    rows: list[dict[str, Any]] = []

    for step in result.steps:
        rows.append(
            {
                "phase": task.experiment.phase,
                "experiment": task.experiment.name,
                "start_partition": task.experiment.start_partition,
                "pipeline": task.experiment.pipeline,
                "dataset": task.dataset,
                "size_class": task.size_class,
                "graph_type": task.graph_type,
                "regime": task.regime,
                "instance": task.instance_name,
                "n": task.graph.number_of_nodes(),
                "m": task.graph.number_of_edges(),
                "run": task.run,
                "seed": task.seed,
                "step_index": step.step_index,
                "step_name": step.step_name,
                "score_before": step.score_before,
                "score_after": step.score_after,
                "absolute_improvement": step.absolute_improvement,
                "relative_improvement": step.relative_improvement,
                "num_moves": step.num_moves,
                "num_passes": step.num_passes,
                "runtime": step.runtime,
                "num_clusters_before": step.num_clusters_before,
                "num_clusters_after": step.num_clusters_after,
            }
        )

    return rows


def evaluate_local_search_task(task: LocalSearchTask) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Evaluates one local-search task.

    Args:
        task (LocalSearchTask): Task containing graph, seed, start partition, and pipeline.

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]]]: One raw pipeline result row and one list of step-level rows.
    """
    graph = task.graph
    experiment = task.experiment

    start_algorithm = build_partition_algorithm(experiment.start_partition)

    start_time = time.perf_counter()
    start_partition = start_algorithm(graph)
    start_runtime = time.perf_counter() - start_time

    start_density = partition_density(graph, start_partition)
    start_stats = partition_stats(start_partition)

    ls_time = time.perf_counter()
    ls_result = run_local_search_pipeline(graph, start_partition, experiment.pipeline)
    ls_runtime = time.perf_counter() - ls_time

    final_partition = ls_result.partition
    final_density = partition_density(graph, final_partition)
    final_stats = partition_stats(final_partition)

    raw_row = {
        "phase": experiment.phase,
        "experiment": experiment.name,
        "start_partition": experiment.start_partition,
        "pipeline": experiment.pipeline,
        "dataset": task.dataset,
        "size_class": task.size_class,
        "graph_type": task.graph_type,
        "regime": task.regime,
        "instance": task.instance_name,
        "n": graph.number_of_nodes(),
        "m": graph.number_of_edges(),
        "edge_density": edge_density(graph),
        "run": task.run,
        "seed": task.seed,
        "start_density": start_density,
        "final_density": final_density,
        "absolute_improvement": final_density - start_density,
        "relative_improvement": 0.0 if start_density == 0 else (final_density - start_density) / start_density,
        "improved": final_density > start_density,
        "num_moves": ls_result.num_moves,
        "num_passes": ls_result.num_passes,
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

    return raw_row, step_rows_from_pipeline_result(task, ls_result)


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

    for dataset in build_datasets(data_root):
        if not dataset.path.exists():
            raise FileNotFoundError(f"Dataset directory {dataset} does not exist.")

        instances = load_instances_json(dataset.path)

        for run_index in range(runs):
            seed = base_seed + 10_000 * run_index

            for instance in instances:
                for experiment in experiments:
                    tasks.append(
                        LocalSearchTask(
                            dataset.name,
                            dataset.size_class,
                            dataset.graph_type,
                            dataset.regime,
                            instance.name,
                            instance.graph,
                            run_index,
                            seed,
                            experiment
                        )
                    )

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
    raw_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    if workers <= 1:
        for index, task in enumerate(tasks):
            raw_row, task_step_rows = evaluate_local_search_task(task)
            raw_rows.append(raw_row)
            step_rows.extend(task_step_rows)

            print(f"[{index}/{len(tasks)}] {task.experiment.name} | {task.instance_name}")

        return raw_rows, step_rows

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_local_search_task, task): task for task in tasks}

        for index, future in enumerate(as_completed(futures)):
            task = futures[future]
            raw_row, task_step_rows = future.result()

            raw_rows.append(raw_row)
            step_rows.extend(task_step_rows)

            print(f"[{index}/{len(tasks)}] {task.experiment.name} | {task.instance_name}")

    return raw_rows, step_rows


def add_relative_scores(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    Adds relative final scores and ranks to local-search results.

    Args:
        raw_results (pd.DataFrame): Raw pipeline result table.

    Returns:
        pd.DataFrame: Result table with relative scores, best flags, and ranks.
    """
    results = raw_results.copy()

    group_columns = ["phase", "dataset", "instance", "run"]
    best_by_instance = results.groupby(group_columns)["final_density"].transform("max")

    results["relative_to_best"] = results["final_density"] / best_by_instance
    results["is_best"] = results["final_density"] == best_by_instance
    results["rank"] = results.groupby(group_columns)["final_density"].rank(method="min", ascending=False,)

    return results


def summarize_pipeline_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates complete local-search pipeline results.

    Args:
        raw_results (pd.DataFrame): Raw pipeline result table.

    Returns:
        pd.DataFrame: Aggregated summary table.
    """
    results = add_relative_scores(raw_results)

    return (
        results.groupby(["phase", "size_class", "graph_type", "regime", "start_partition", "pipeline", "experiment"])
        .agg(
            observations=("instance", "count"),
            mean_start_density=("start_density", "mean"),
            mean_final_density=("final_density", "mean"),
            mean_absolute_improvement=("absolute_improvement", "mean"),
            mean_relative_improvement=("relative_improvement", "mean"),
            improvement_rate=("improved", "mean"),
            mean_relative_to_best=("relative_to_best", "mean"),
            mean_rank=("rank", "mean"),
            wins=("is_best", "sum"),
            mean_num_moves=("num_moves", "mean"),
            mean_num_passes=("num_passes", "mean"),
            mean_start_runtime=("start_runtime", "mean"),
            mean_ls_runtime=("ls_runtime", "mean"),
            mean_total_runtime=("total_runtime", "mean"),
            mean_start_num_clusters=("start_num_clusters", "mean"),
            mean_final_num_clusters=("final_num_clusters", "mean"),
        )
        .reset_index()
        .sort_values(["phase", "size_class", "graph_type", "regime", "mean_rank"])
    )


def summarize_step_results(step_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates local-search step results.

    Args:
        step_results (pd.DataFrame): Step-level result table.

    Returns:
        pd.DataFrame: Aggregated step summary table.
    """
    return (
        step_results.groupby(["phase", "size_class", "graph_type", "regime", "start_partition", "pipeline", "step_index", "step_name"])
        .agg(
            observations=("instance", "count"),
            mean_score_before=("score_before", "mean"),
            mean_score_after=("score_after", "mean"),
            mean_absolute_improvement=("absolute_improvement", "mean"),
            mean_relative_improvement=("relative_improvement", "mean"),
            active_rate=("num_moves", lambda values: (values > 0).mean()),
            mean_num_moves=("num_moves", "mean"),
            median_num_moves=("num_moves", "median"),
            mean_num_passes=("num_passes", "mean"),
            median_num_passes=("num_passes", "median"),
            mean_runtime=("runtime", "mean"),
            mean_num_clusters_before=("num_clusters_before", "mean"),
            mean_num_clusters_after=("num_clusters_after", "mean"),
        )
        .reset_index()
        .sort_values(["phase", "size_class", "graph_type", "regime", "start_partition", "pipeline", "step_index"])
    )


def write_local_search_results(
        raw_results: pd.DataFrame,
        step_results: pd.DataFrame,
        results_dir: Path,
) -> None:
    """
    Writes local-search results to CSV files.
    Both complete result tables and phase-separated result tables are written.

    Args:
        raw_results (pd.DataFrame): Raw pipeline result table.
        step_results (pd.DataFrame): Step-level result table.
        results_dir (Path): Output directory.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_results = add_relative_scores(raw_results)

    rounded_for_export(raw_results).to_csv(results_dir / "raw_results.csv", index=False)
    rounded_for_export(step_results).to_csv(results_dir / "step_results.csv", index=False)
    rounded_for_export(summarize_pipeline_results(raw_results)).to_csv(results_dir / "summary.csv", index=False)
    rounded_for_export(summarize_step_results(step_results)).to_csv(results_dir / "step_summary.csv", index=False)

    for phase in sorted(raw_results["phase"].unique()):
        phase_dir = results_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)

        phase_raw = raw_results[raw_results["phase"] == phase].copy()
        phase_steps = step_results[step_results["phase"] == phase].copy()

        rounded_for_export(phase_raw).to_csv(phase_dir / "raw_results.csv", index=False)
        rounded_for_export(phase_steps).to_csv(phase_dir / "step_results.csv", index=False)
        rounded_for_export(summarize_pipeline_results(phase_raw)).to_csv(phase_dir / "summary.csv", index=False)
        rounded_for_export(summarize_step_results(phase_steps)).to_csv(phase_dir / "step_summary.csv", index=False)