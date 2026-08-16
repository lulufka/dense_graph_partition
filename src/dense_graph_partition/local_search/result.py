from dataclasses import dataclass

import networkx as nx

from dense_graph_partition.core.evaluation import partition_density
from dense_graph_partition.core.types import Partition
from dense_graph_partition.local_search.state import PartitionState, state_to_partition


@dataclass(frozen=True)
class LocalSearchResult:
    """
    Stores the result and basic statistics of a local-search run.

    Attributes:
        partition (Partition): Final partition resulting from the local-search run.
        initial_score (float): Partition density before local search.
        final_score (float): Partition density after local search.
        num_moves (int): Number of accepted local-search operations.
        num_passes (int): Number of completed passes or iterations.
        num_improving_moves (int): Number of moves which improve the partition density.
        num_zero_gain_moves (int): Number of plateau moves.
    """
    partition: Partition
    initial_score: float
    final_score: float
    num_moves: int
    num_passes: int
    num_improving_moves: int = 0
    num_zero_gain_moves: int = 0

    @property
    def num_operations(self) -> int:
        """
        Returns the number of accepted local-search operations.
        """
        return self.num_moves

    @property
    def absolute_improvement(self) -> float:
        """
        Returns the absolute improvement in partition density.
        """
        return self.final_score - self.initial_score

    @property
    def relative_improvement(self) -> float:
        """
        Returns the relative improvement in partition density.
        """
        if self.initial_score == 0:
            return 0.0

        return self.absolute_improvement / self.initial_score

    @property
    def improved(self) -> bool:
        """
        Returns whether local search improved the initial partition density.
        """
        return self.final_score > self.initial_score


def build_local_search_result(
        G: nx.Graph,
        state: PartitionState,
        initial_score: float,
        num_moves: int,
        num_passes: int,
        num_improving_moves: int = 0,
        num_zero_gain_moves: int = 0,
) -> LocalSearchResult:
    final_partition = state_to_partition(state)
    final_score = partition_density(G, final_partition)

    return LocalSearchResult(
        partition=final_partition,
        initial_score=initial_score,
        final_score=final_score,
        num_moves=num_moves,
        num_passes=num_passes,
        num_improving_moves=num_improving_moves,
        num_zero_gain_moves=num_zero_gain_moves,
    )