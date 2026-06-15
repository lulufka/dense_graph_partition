import networkx as nx

from dense_graph_partition.core.evaluation import partition_density
from dense_graph_partition.local_search.result import LocalSearchResult
from dense_graph_partition.local_search.state import PartitionState, state_to_partition


def build_local_search_result(G: nx.Graph, state: PartitionState, initial_score: float, num_moves: int, num_passes: int) -> LocalSearchResult:
    final_partition = state_to_partition(state)
    final_score = partition_density(G, final_partition)
    return LocalSearchResult(final_partition, num_moves, num_passes, initial_score, final_score)

