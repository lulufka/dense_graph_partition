import random
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import networkx as nx

GraphGenerator = Callable[..., nx.Graph]


@dataclass(frozen=True)
class InstanceSpec:
    graph_type: str
    regime: str
    size_class: str


@dataclass(frozen=True)
class GenerationTask:
    output_dir: Path
    graph_type: str
    regime: str
    size_class: str
    index: int
    n: int
    seed: int

@dataclass(frozen=True)
class GroundTruthInstanceSpec(InstanceSpec):
    community_size_class: str


@dataclass(frozen=True)
class GroundTruthGenerationTask(GenerationTask):
    community_size_class: str


SIZE_RANGES = {
    "small": (50, 250),
    "large": (500, 1500),
}

COMMUNITY_SIZE_FRACTIONS = {
    "small": 0.05,
    "large": 0.20,
}


def sample_sizes(count: int, n_min: int, n_max: int, seed: int) -> list[int]:
    """
    Samples and sorts graph sizes.

    Args:
        count (int): Number of graph sizes to sample.
        n_min (int): Minimum graph size.
        n_max (int): Maximum graph size.
        seed (int): Random seed used for reproducibility.

    Returns:
        list[int]: Sorted sampled graph sizes.
    """
    rng = random.Random(seed)

    sizes = [rng.randint(n_min, n_max) for _ in range(count)]
    sizes.sort()

    return sizes


def sample_sizes_by_class(count: int, seed: int) -> dict[str, list[int]]:
    """
    Samples graph sizes for all supported size classes.

    Args:
        count (int): Number of graph sizes per size class.
        seed (int): Base seed used for reproducibility.

    Returns:
        dict[str, list[int]]: Sampled graph sizes by size class.
    """
    small_min, small_max = SIZE_RANGES["small"]
    large_min, large_max = SIZE_RANGES["large"]

    return {
        "small": sample_sizes(count=count, n_min=small_min, n_max=small_max, seed=seed),
        "large": sample_sizes(count=count, n_min=large_min, n_max=large_max, seed=seed + 1),
    }


def target_average_degree(n: int, regime: str) -> float:
    """
    Returns the target average node degree for a graph instance.

    Args:
        n (int): Number of nodes in the graph.
        regime (str): Density regime. Supported values are ``"sparse"`` and ``"dense"``.

    Returns:
        float: Target average node degree.
    """
    if regime == "sparse":
        return 8.0

    if regime == "dense":
        return max(16.0, 0.04 * n)

    raise ValueError(f"Unknown regime: {regime}")


def target_community_size(n: int, community_size_class: str) -> int:
    """
    Returns the target community size for a graph.
    Small communities contain approximately 5 percent of all vertices, while large communities contain approximately 20 percent.

    Args:
        n (int): Number of vertices.
        community_size_class (str): Supported values are ``"small"`` and ``"large"``.

    Returns:
        int: Target community size.
    """
    try:
        fraction = COMMUNITY_SIZE_FRACTIONS[community_size_class]
    except KeyError as exc:
        raise ValueError(f"Unknown community size class: {community_size_class}") from exc

    return max(3, round(fraction * n))


def generate_connected_instance(generator: GraphGenerator, seed: int, max_attempts: int = 1000, **generator_kwargs: object) -> nx.Graph:
    """
    Generates a connected graph instance.
    The graph generator is repeatedly called with consecutive seeds until a connected graph is produced.

    Args:
        generator (GraphGenerator): Graph generator function.
        seed (int): Initial random seed.
        max_attempts (int): Maximum number of generation attempts.
        **generator_kwargs (object): Additional arguments passed to the graph generator.

    Returns:
        nx.Graph: Connected graph instance.
    """
    for attempt in range(max_attempts):
        attempt_seed = seed + attempt

        graph = generator(seed=attempt_seed, **generator_kwargs)

        if nx.is_connected(graph):
            return graph

    raise RuntimeError(f"Could not generate a connected graph after {max_attempts} attempts")