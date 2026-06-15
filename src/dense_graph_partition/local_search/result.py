from dataclasses import dataclass

from dense_graph_partition.core.types import Partition


@dataclass(frozen=True)
class LocalSearchResult:
    """
    Stores the result and basic statistic of a local search run.

    Attributes:
        partition (Partition): Final partition resulting from the local search.
        num_moves (int): Number of accepted local search operations.
        num_passes (int): Number of completed passes or iterations.
        initial_score (float): Partition density before the local search.
        final_score (float): Partition density after the local search.
    """
    partition: Partition
    num_moves: int
    num_passes: int
    initial_score: float
    final_score: float

    @property
    def absolut_improvement(self) -> float:
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

        return self.absolut_improvement / self.initial_score

    @property
    def improved(self) -> bool:
        """
        Returns whether local search improved the initial partition.
        """
        return self.final_score > self.initial_score