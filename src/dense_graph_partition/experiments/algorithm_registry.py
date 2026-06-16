from collections.abc import Callable
from functools import partial

import networkx as nx

from dense_graph_partition.adapters.kapoce import kapoce_partition
from dense_graph_partition.adapters.leiden import leiden_mdgp_partition
from dense_graph_partition.algorithms.basic import singleton_partition
from dense_graph_partition.algorithms.matching import (
    high_degree_first_matching_partition,
    high_degree_product_matching_partition,
    matching_partition,
    maximum_matching_partition,
)
from dense_graph_partition.config import load_kapoce_config
from dense_graph_partition.core.types import Partition
from dense_graph_partition.experiments.baseline_runner import AlgorithmSpec


StartAlgorithm = Callable[[nx.Graph], Partition]


BASELINE_ALGORITHM_NAMES = [
    "matching",
    "maximum_matching",
    "high_degree_first_matching",
    "high_degree_product_matching",
    "leiden_mdgp",
    "kapoce",
]


START_PARTITION_NAMES = [
    "singleton",
    "matching",
    "maximum_matching",
    "high_degree_first_matching",
    "high_degree_product_matching",
    "leiden_mdgp",
    "kapoce",
]


def build_partition_algorithm(name: str) -> StartAlgorithm:
    """
    Builds a partitioning algorithm by name.

    Args:
        name (str): Algorithm name.

    Returns:
        StartAlgorithm: Callable that computes a partition.

    Raises:
        ValueError: If the algorithm name is unknown.
    """
    if name == "singleton":
        return singleton_partition

    if name == "matching":
        return matching_partition

    if name == "maximum_matching":
        return maximum_matching_partition

    if name == "high_degree_first_matching":
        return high_degree_first_matching_partition

    if name == "high_degree_product_matching":
        return high_degree_product_matching_partition

    if name == "leiden_mdgp":
        return leiden_mdgp_partition

    if name == "kapoce":
        kapoce_config = load_kapoce_config()
        return partial(kapoce_partition, executable_path=kapoce_config.executable_path, config_path=kapoce_config.config_path)

    raise ValueError(f"Unknown partition algorithm: {name}")


def build_algorithm_specs(names: list[str]) -> list[AlgorithmSpec]:
    """
    Builds AlgorithmSpec objects for experiment runners.

    Args:
        names (list[str]): Algorithm names.

    Returns:
        list[AlgorithmSpec]: Algorithm specifications.
    """
    return [AlgorithmSpec(name=name, run=build_partition_algorithm(name)) for name in names]


def build_baseline_algorithm_specs() -> list[AlgorithmSpec]:
    """
    Builds the baseline algorithms for experiment 1.

    Returns:
        list[AlgorithmSpec]: Baseline algorithm specifications.
    """
    return build_algorithm_specs(BASELINE_ALGORITHM_NAMES)
