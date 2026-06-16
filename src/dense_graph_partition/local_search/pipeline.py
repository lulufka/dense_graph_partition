import time
from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx

from dense_graph_partition.core.evaluation import partition_density, partition_num_clusters
from dense_graph_partition.core.types import Partition
from dense_graph_partition.local_search.merge import refine_partition_merge_first, refine_partition_merge_best
from dense_graph_partition.local_search.move import refine_partition_move_first, refine_partition_move_best, \
    refine_partition_move_plateau
from dense_graph_partition.local_search.peel import refine_partition_peel_node
from dense_graph_partition.local_search.result import LocalSearchResult
from dense_graph_partition.local_search.split import refine_partition_split_min_cut, refine_partition_bridge_split

LocalSearchRefiner = Callable[[nx.Graph, Partition], LocalSearchResult]


@dataclass(frozen=True)
class PipelineStepResult:
    """
   Stores the result of one local-search step inside a pipeline.

   Attributes:
       step_index (int): Position of the step in the pipeline.
       step_name (str): Name of the local-search refiner.
       score_before (float): Partition density before this step.
       score_after (float): Partition density after this step.
       num_moves (int): Number of accepted operations in this step.
       num_passes (int): Number of passes or iterations used by this step.
       runtime (float): Runtime of this step in seconds.
       num_clusters_before (int): Number of clusters before this step.
       num_clusters_after (int): Number of clusters after this step.
   """
    step_index: int
    step_name: str
    score_before: float
    score_after: float
    num_moves: int
    num_passes: int
    runtime: float
    num_clusters_before: int
    num_clusters_after: int

    @property
    def absolute_improvement(self) -> float:
        return self.score_after - self.score_before

    @property
    def relative_improvement(self) -> float:
        if self.score_before == 0:
            return 0.0
        return self.absolute_improvement / self.score_before


@dataclass(frozen=True)
class PipelineResult:
    """
    Stores the result of a local-search pipeline.

    Attributes:
        partition (Partition): Final partition after all pipeline steps.
        num_moves (int): Total number of accepted local-search operations.
        num_passes (int): Total number of passes over all pipeline steps.
        initial_score (float): Score before the first pipeline step.
        final_score (float): Score after the last pipeline step.
        steps (list[PipelineStepResult]): Per-step results of the pipeline.
    """
    partition: Partition
    initial_score: float
    final_score: float
    num_moves: int
    num_passes: int
    steps: list[PipelineStepResult]

    @property
    def absolute_improvement(self) -> float:
        return self.final_score - self.initial_score

    @property
    def relative_improvement(self) -> float:
        if self.initial_score == 0:
            return 0.0
        return self.absolute_improvement / self.initial_score

    @property
    def improved(self) -> bool:
        return self.final_score > self.initial_score


LOCAL_SEARCH_REFINERS: dict[str, LocalSearchRefiner] = {
    "move_first": refine_partition_move_first,
    "move_best": refine_partition_move_best,
    "move_plateau": refine_partition_move_plateau,
    "merge_first": refine_partition_merge_first,
    "merge_best": refine_partition_merge_best,
    "split_min_cut": refine_partition_split_min_cut,
    "bridge_split": refine_partition_bridge_split,
    "peel_node": refine_partition_peel_node,
}


def parse_pipeline(pipeline: str) -> list[str]:
    """
    Parses a comma-separated pipeline string.

    Args:
        pipeline (str): Comma-separated local-search step names.

    Returns:
        list[str]: Pipeline step names.

    Raises:
        ValueError: If the pipeline is empty or contains unknown steps.
    """
    steps = [step.strip() for step in pipeline.split(",") if step.strip()]

    if not steps:
        raise ValueError("Pipeline must contain at least one local-search step.")

    unknown = [step for step in steps if step not in LOCAL_SEARCH_REFINERS]
    if unknown:
        known = ", ".join(sorted(LOCAL_SEARCH_REFINERS))
        raise ValueError(f"Unknown local-search steps: {unknown}. Known steps: {known}.")

    return steps


def run_local_search_pipeline(G: nx.Graph, partition: Partition, pipeline: str) -> PipelineResult:
    """
    Runs a sequence of local-search refiners.

    Args:
        G (nx.Graph): Input graph.
        partition (Partition): Initial partition.
        pipeline (str): Comma-separated local-search step names.

    Returns:
        PipelineResult: Final partition and aggregated local-search statistics.
    """
    steps = parse_pipeline(pipeline)

    current_partition = partition
    initial_score = partition_density(G, current_partition)
    total_moves = 0
    total_passes = 0
    step_results: list[PipelineStepResult] = []

    for step_index, step_name in enumerate(steps):
        refiner = LOCAL_SEARCH_REFINERS[step_name]

        score_before = partition_density(G, current_partition)
        clusters_before = partition_num_clusters(current_partition)

        start_time = time.perf_counter()
        result = refiner(G, current_partition)
        runtime = time.perf_counter() - start_time

        score_after = result.final_score
        clusters_after = partition_num_clusters(result.partition)

        step_results.append(
            PipelineStepResult(step_index, step_name, score_before, score_after, result.num_moves, result.num_passes, runtime, clusters_before, clusters_after)
        )

        current_partition = result.partition
        total_moves += result.num_moves
        total_passes += result.num_passes

    final_score = partition_density(G, current_partition)

    return PipelineResult(current_partition, initial_score, final_score, total_moves, total_passes, step_results)