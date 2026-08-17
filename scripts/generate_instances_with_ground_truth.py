import argparse
import random
from pathlib import Path

import networkx as nx

from dense_graph_partition.core.graph_io import save_ground_truth_graph_json
from dense_graph_partition.core.types import Partition
from dense_graph_partition.experiments.data_generation import GroundTruthGenerationTask, GroundTruthInstanceSpec, sample_sizes_by_class, \
    target_average_degree, target_community_size
from dense_graph_partition.experiments.run_tasks import run_tasks


DEFAULT_SPECS = [
    GroundTruthInstanceSpec(
        graph_type=graph_type,
        regime=regime,
        size_class=size_class,
        community_size_class=community_size_class,
    )
    for graph_type in (
        "planted_partition",
        "gaussian_partition",
        "random_partition",
    )
    for regime in ("sparse", "dense")
    for size_class in ("small", "large")
    for community_size_class in ("small", "large")
]


IN_OUT_RATIO = 10.0
GAUSSIAN_VARIANCE_FACTOR = 2.0


def average_internal_neighbors(sizes: list[int]) -> float:
    """
    Computes the average number of possible same-community neighbors of a uniformly selected vertex.
    For community sizes s_1, ..., s_k this is sum_i s_i * (s_i - 1) / n.

    Args:
        sizes (list[int]): Community sizes.

    Returns:
        float: Average number of possible intra-community neighbors.
    """
    n = sum(sizes)

    return sum(size * (size - 1) for size in sizes) / n


def connection_probabilities(n: int, target_degree: float, internal_neighbors: float, ratio: float = IN_OUT_RATIO) -> tuple[float, float]:
    """
    Computes p_in and p_out for a target expected average degree.

    The probabilities are chosen such that p_in is ratio times p_out whenever possible.
    Together with E[d] = internal_neighbors * p_in + external_neighbors * p_out this determines both probabilities.
    If the resulting p_in would exceed 1, p_in is fixed to 1 and p_out is chosen to preserve the target expected degree.

    Args:
        n (int): Number of vertices.
        target_degree (float): Desired average degree.
        internal_neighbors (float): Average number of possible same-community neighbors.
        ratio (float): Desired ratio p_in / p_out.

    Returns:
        tuple[float, float]: ``(p_in, p_out)``.
    """
    external_neighbors = (n - 1) - internal_neighbors

    denominator = ratio * internal_neighbors + external_neighbors

    p_out = target_degree / denominator
    p_in = ratio * p_out

    if p_in <= 1.0:
        return p_in, p_out

    p_in = 1.0

    if external_neighbors == 0:
        return p_in, 0.0

    p_out = (target_degree - internal_neighbors) / external_neighbors
    p_out = min(1.0, max(0.0, p_out))

    return p_in, p_out


def random_partition_sizes(n: int, target_size: int, seed: int) -> list[int]:
    """
    Generates heterogeneous community sizes around a target size.

    Each community size is sampled uniformly between 50 and 150 percent of the target size. The final community receives the remaining vertices.

    Args:
        n (int): Number of vertices.
        target_size (int): Desired average community size.
        seed (int): Random seed.

    Returns:
        list[int]: Community sizes summing to n.
    """
    rng = random.Random(seed)

    minimum = max(2, round(0.5 * target_size))
    maximum = max(minimum, round(1.5 * target_size))

    sizes: list[int] = []
    remaining = n

    while remaining > 0:
        if remaining <= maximum:
            if remaining < minimum and sizes:
                sizes[-1] += remaining
            else:
                sizes.append(remaining)

            break

        size = rng.randint(minimum, maximum)

        if 0 < remaining - size < minimum:
            size = remaining

        sizes.append(size)
        remaining -= size

    return sizes


def generate_planted_partition_graph(n: int, seed: int, regime: str, community_size_class: str) -> nx.Graph:
    """
    Generates a planted partition graph with equally sized communities.
    Since NetworkX requires n = l * k, the requested number of nodes may be adjusted slightly.

    Args:
        n (int): Approximate number of vertices.
        seed (int): Random seed.
        regime (str): Density regime.
        community_size_class (str): Community size regime.

    Returns:
        nx.Graph: Generated graph including the partition graph attribute.
    """
    target_size = target_community_size(n, community_size_class)

    num_communities = max(2, round(n / target_size))
    community_size = max(2, round(n / num_communities))

    actual_n = num_communities * community_size

    sizes = [community_size] * num_communities

    internal_neighbors = average_internal_neighbors(sizes)

    p_in, p_out = connection_probabilities(
        n=actual_n,
        target_degree=target_average_degree(actual_n, regime),
        internal_neighbors=internal_neighbors,
    )

    graph = nx.planted_partition_graph(
        l=num_communities,
        k=community_size,
        p_in=p_in,
        p_out=p_out,
        seed=seed,
    )

    graph.graph["generation_metadata"] = {
        "p_in": p_in,
        "p_out": p_out,
        "target_average_degree": target_average_degree(actual_n, regime),
        "target_community_size": target_size,
    }

    return graph


def generate_random_partition_graph(n: int, seed: int, regime: str, community_size_class: str) -> nx.Graph:
    """
    Generates a random partition graph with heterogeneous community sizes.

    Args:
        n (int): Number of vertices.
        seed (int): Random seed.
        regime (str): Density regime.
        community_size_class (str): Community size regime.

    Returns:
        nx.Graph: Generated graph including the partition graph attribute.
    """
    target_size = target_community_size(n, community_size_class)

    sizes = random_partition_sizes(n=n, target_size=target_size, seed=seed)

    internal_neighbors = average_internal_neighbors(sizes)

    p_in, p_out = connection_probabilities(
        n=n,
        target_degree=target_average_degree(n, regime),
        internal_neighbors=internal_neighbors,
    )

    graph = nx.random_partition_graph(
        sizes=sizes,
        p_in=p_in,
        p_out=p_out,
        seed=seed,
    )

    graph.graph["generation_metadata"] = {
        "p_in": p_in,
        "p_out": p_out,
        "target_average_degree": target_average_degree(n, regime),
        "target_community_size": target_size,
    }

    return graph


def generate_gaussian_partition_graph(n: int, seed: int, regime: str, community_size_class: str) -> nx.Graph:
    """
    Generates a Gaussian random partition graph.

    Community sizes are normally distributed around the target community size. NetworkX uses variance s / v.

    Args:
        n (int): Number of vertices.
        seed (int): Random seed.
        regime (str): Density regime.
        community_size_class (str): Community size regime.

    Returns:
        nx.Graph: Generated graph including the partition graph attribute.
    """
    target_size = target_community_size(n, community_size_class)

    s = float(target_size)
    v = GAUSSIAN_VARIANCE_FACTOR

    internal_neighbors = max(1.0, s - 1.0 + 1.0 / v)

    p_in, p_out = connection_probabilities(
        n=n,
        target_degree=target_average_degree(n, regime),
        internal_neighbors=internal_neighbors,
    )

    graph = nx.gaussian_random_partition_graph(
        n=n,
        s=s,
        v=v,
        p_in=p_in,
        p_out=p_out,
        seed=seed,
    )

    graph.graph["generation_metadata"] = {
        "p_in": p_in,
        "p_out": p_out,
        "target_average_degree": target_average_degree(n, regime),
        "target_community_size": target_size,
        "gaussian_s": s,
        "gaussian_v": v,
    }

    return graph


GENERATORS = {
    "planted_partition": generate_planted_partition_graph,
    "gaussian_partition": generate_gaussian_partition_graph,
    "random_partition": generate_random_partition_graph,
}


def extract_ground_truth(graph: nx.Graph) -> Partition:
    """
    Extracts the ground-truth partition stored by NetworkX.

    Args:
        graph (nx.Graph): Generated partition graph.

    Returns:
        Partition: Ground-truth communities.
    """
    partition = graph.graph.get("partition")

    if partition is None:
        raise ValueError("Generated graph does not contain a 'partition' graph attribute")

    return [set(community) for community in partition]


def generate_connected_ground_truth_instance(task: GroundTruthGenerationTask, max_attempts: int = 1000) -> nx.Graph:
    """
    Generates a connected ground-truth graph.

    Args:
        task (GroundTruthGenerationTask): Generation task.
        max_attempts (int): Maximum number of generation attempts.

    Returns:
        nx.Graph: Connected generated graph.
    """
    generator = GENERATORS[task.graph_type]

    for attempt in range(max_attempts):
        graph = generator(
            n=task.n,
            seed=task.seed + attempt,
            regime=task.regime,
            community_size_class=task.community_size_class,
        )

        if nx.is_connected(graph):
            return graph

    raise RuntimeError(f"Could not generate connected instance after {max_attempts} attempts: {task}")


def generate_instance_task(task: GroundTruthGenerationTask) -> str:
    """
    Generates and saves one ground-truth graph instance.

    Args:
        task (GroundTruthGenerationTask): Instance description.

    Returns:
        str: Generated instance name.
    """
    graph = generate_connected_ground_truth_instance(task)

    ground_truth = extract_ground_truth(graph)

    actual_n = graph.number_of_nodes()

    name = (
        f"{task.size_class}_{task.graph_type}_{task.regime}_communities-{task.community_size_class}_{task.index:03d}_n{actual_n}"
    )

    target_dir = task.output_dir / task.size_class / task.graph_type / task.regime / f"communities_{task.community_size_class}"


    metadata = {
        "graph_type": task.graph_type,
        "regime": task.regime,
        "size_class": task.size_class,
        "community_size_class": task.community_size_class,
        "num_communities": len(ground_truth),
        "community_sizes": sorted(len(cluster) for cluster in ground_truth),
        **graph.graph.get("generation_metadata", {}),
    }

    save_ground_truth_graph_json(
        G=graph,
        path=target_dir / f"{name}.json",
        name=name,
        ground_truth=ground_truth,
        metadata=metadata,
    )

    return name


def build_generation_tasks(output_dir: Path, specs: list[GroundTruthInstanceSpec], sizes_by_class: dict[str, list[int]], base_seed: int) -> list[GroundTruthGenerationTask]:
    """
    Builds all ground-truth graph generation tasks.

    Args:
        output_dir (Path): Root output directory.
        specs (list[GroundTruthInstanceSpec]): Generation configurations.
        sizes_by_class (dict[str, list[int]]): Graph sizes by size class.
        base_seed (int): Base random seed.

    Returns:
        list[GroundTruthGenerationTask]: Generation tasks.
    """
    tasks: list[GroundTruthGenerationTask] = []

    for spec_index, spec in enumerate(specs):
        sizes = sizes_by_class[spec.size_class]

        spec_seed = base_seed + 10_000_000 * spec_index

        for index, n in enumerate(sizes):
            tasks.append(
                GroundTruthGenerationTask(
                    output_dir=output_dir,
                    graph_type=spec.graph_type,
                    regime=spec.regime,
                    size_class=spec.size_class,
                    community_size_class=spec.community_size_class,
                    index=index,
                    n=n,
                    seed=spec_seed + index * 10_000,
                )
            )

    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ground-truth community instances.")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ground_truth"),
        help="Directory where generated instances are stored.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base seed for reproducible instance generation.",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=250,
        help="Number of instances per configuration.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of worker processes. Use 1 for sequential execution.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sizes_by_class = sample_sizes_by_class(count=args.count, seed=args.seed)

    tasks = build_generation_tasks(
        output_dir=args.output_dir,
        specs=DEFAULT_SPECS,
        sizes_by_class=sizes_by_class,
        base_seed=args.seed,
    )

    print(f"Prepared {len(tasks)} ground-truth generation tasks.", flush=True)

    run_tasks(
        tasks=tasks,
        evaluate=generate_instance_task,
        workers=args.workers,
        describe=lambda task: (
            f"{task.graph_type} | "
            f"{task.regime} | "
            f"{task.size_class} | "
            f"communities={task.community_size_class} | "
            f"n={task.n}"
        ),
    )


if __name__ == "__main__":
    main()