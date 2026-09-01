import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from dense_graph_partition.core.graph_io import save_ground_truth_graph_json
from dense_graph_partition.core.types import Partition
from dense_graph_partition.experiments.data_generation import target_average_degree, target_community_size
from dense_graph_partition.experiments.run_tasks import run_tasks


SIZE_CLASSES = ["small", "large"]
REGIMES = ["sparse", "dense"]
COMMUNITY_SIZE_CLASSES = ["small", "large"]
LARGE_COMMUNITY_SIZE_CLASSES = ["tiny", "small", "large"]
NOISE_LEVELS = [0.05, 0.2]

GAUSSIAN_VARIANCE_FACTOR = 2.0

GROUND_TRUTH_SIZE_RANGES = {
    "small": (175, 250),
    "large": (500, 1500),
}


@dataclass(frozen=True)
class GroundTruthGenerationTask:
    """
    Describes one Gaussian ground-truth graph generation task.

    Attributes:
        output_dir (Path): Root output directory.
        size_class (str): Graph size category.
        regime (str): Intra-community density regime.
        community_size_class (str): Community size category.
        noise (float): Expected fraction of external neighbors.
        index (int): Instance index within the configuration.
        n (int): Number of vertices.
        seed (int): Random seed.
    """

    output_dir: Path
    size_class: str
    regime: str
    community_size_class: str
    noise: float
    index: int
    n: int
    seed: int


def connection_probabilities(n: int, target_degree: float, internal_neighbors: float, noise: float) -> tuple[float, float]:
    """
    Computes intra- and inter-community edge probabilities.
    The noise parameter specifies the expected fraction of a vertex's neighbors that lie outside its ground-truth community.
    Thus,

        expected_internal_degree = (1 - noise) * target_degree
        expected_external_degree = noise * target_degree

    Args:
        n (int): Number of vertices.
        target_degree (float): Desired expected average degree.
        internal_neighbors (float): Expected number of possible same-community neighbors.
        noise (float): Expected fraction of external neighbors.

    Returns:
        tuple[float, float]: Intra- and inter-community edge probabilities ``(p_in, p_out)``.
    """
    if not 0.0 <= noise < 1.0:
        raise ValueError(f"Noise must satisfy 0 <= noise < 1, got {noise}.")

    external_neighbors = (n - 1) - internal_neighbors

    expected_internal_degree = (1.0 - noise) * target_degree
    expected_external_degree = noise * target_degree

    p_in = expected_internal_degree / internal_neighbors
    p_out = expected_external_degree / external_neighbors


    if p_in > 1.0:
        raise ValueError(
            f"Requested combination is infeasible: n={n}, target_degree={target_degree:.2f}, internal_neighbors={internal_neighbors:.2f}, "
            f"noise={noise:.2f}, p_in={p_in:.3f}."
        )

    if p_out > 1.0:
        raise ValueError(
            f"Requested combination is infeasible: n={n}, target_degree={target_degree:.2f}, external_neighbors={external_neighbors:.2f}, "
            f"noise={noise:.2f}, p_out={p_out:.3f}."
        )

    return p_in, p_out



def generate_gaussian_partition_graph(n: int, seed: int, regime: str, community_size_class: str, noise: float) -> nx.Graph:
    """
    Generates a Gaussian random partition graph.

    Args:
        n (int): Number of vertices.
        seed (int): Random seed.
        regime (str): Intra-community density regime.
        community_size_class (str): Community size category.
        noise (float): Expected fraction of external neighbors.

    Returns:
        nx.Graph: Generated graph including its ground-truth partition.
    """
    target_size = target_community_size(n, community_size_class, regime)

    s = float(target_size)
    v = GAUSSIAN_VARIANCE_FACTOR

    internal_neighbors = s + 1 / v - 1

    target_degree = target_average_degree(n, regime)

    p_in, p_out = connection_probabilities(n=n, target_degree=target_degree, internal_neighbors=internal_neighbors, noise=noise)

    graph = nx.gaussian_random_partition_graph(
        n=n,
        s=s,
        v=v,
        p_in=p_in,
        p_out=p_out,
        seed=seed,
    )

    graph.graph["generation_metadata"] = {
        "regime": regime,
        "noise": noise,
        "p_in": p_in,
        "p_out": p_out,
        "target_community_size": target_size,
        "gaussian_s": s,
        "gaussian_v": v,
    }

    return graph


def extract_ground_truth(graph: nx.Graph) -> Partition:
    """
    Extracts the planted ground-truth partition stored by NetworkX.

    Args:
        graph (nx.Graph): Generated Gaussian partition graph.

    Returns:
        Partition: Ground-truth communities.
    """
    partition = graph.graph.get("partition")

    if partition is None:
        raise ValueError("Generated graph does not contain a 'partition' graph attribute.")

    return [set(community) for community in partition]


def generate_connected_instance(
        n: int,
        regime: str,
        community_size_class: str,
        noise: float,
        seed: int,
        max_attempts: int = 1000,
) -> nx.Graph | None:
    """
    Tries to generate a connected Gaussian ground-truth graph for
    a fixed number of vertices.

    Returns None if no connected instance can be generated within
    the maximum number of attempts.
    """
    for attempt in range(max_attempts):
        graph = generate_gaussian_partition_graph(
            n=n,
            seed=seed + attempt,
            regime=regime,
            community_size_class=community_size_class,
            noise=noise,
        )

        if nx.is_connected(graph):
            return graph

    return None


def noise_name(noise: float) -> str:
    """
    Converts a noise value into a filename-safe representation.
    """
    return f"{noise:g}".replace(".", "-")


def generate_instance_task(
        task: GroundTruthGenerationTask,
) -> str:
    """
    Generates and saves one ground-truth graph instance.

    If no connected graph can be generated for the initially sampled
    graph size, another graph size from the same size class is sampled.
    This is repeated until one connected instance is obtained.

    Args:
        task (GroundTruthGenerationTask): Generation task.

    Returns:
        str: Generated instance name.
    """
    min_n, max_n = GROUND_TRUTH_SIZE_RANGES[
        task.size_class
    ]

    rng = random.Random(task.seed)

    n = task.n
    size_attempt = 0

    while True:
        generation_seed = (
                task.seed
                + size_attempt * 1_000_000
        )

        graph = generate_connected_instance(
            n=n,
            regime=task.regime,
            community_size_class=task.community_size_class,
            noise=task.noise,
            seed=generation_seed,
        )

        if graph is not None:
            break

        print(
            f"\nRETRY WITH NEW SIZE: "
            f"{task.size_class} | "
            f"{task.regime} | "
            f"communities={task.community_size_class} | "
            f"noise={task.noise:g} | "
            f"index={task.index} | "
            f"failed n={n}",
            flush=True,
        )

        n = rng.randint(
            min_n,
            max_n,
        )

        size_attempt += 1

    ground_truth = extract_ground_truth(graph)

    community_sizes = [
        len(cluster)
        for cluster in ground_truth
    ]

    noise = noise_name(task.noise)

    name = (
        f"{task.size_class}_gaussian_partition_"
        f"{task.regime}_"
        f"communities-{task.community_size_class}_"
        f"noise-{noise}_"
        f"{task.index:03d}_n{n}"
    )

    target_dir = (
            task.output_dir
            / task.size_class
            / "gaussian_partition"
            / task.regime
            / f"communities_{task.community_size_class}"
            / f"noise_{noise}"
    )

    metadata = {
        "graph_type": "gaussian_partition",
        "size_class": task.size_class,
        "regime": task.regime,
        "community_size_class": task.community_size_class,
        "noise": task.noise,
        "num_communities": len(ground_truth),
        "community_sizes": sorted(community_sizes),
        "min_community_size": min(community_sizes),
        "max_community_size": max(community_sizes),
        "avg_community_size": (
                sum(community_sizes)
                / len(community_sizes)
        ),
        "size_resampling_attempts": size_attempt,
        "initial_n": task.n,
        **graph.graph.get(
            "generation_metadata",
            {},
        ),
    }

    save_ground_truth_graph_json(
        G=graph,
        path=target_dir / f"{name}.json",
        name=name,
        ground_truth=ground_truth,
        metadata=metadata,
    )

    return name


def sample_ground_truth_sizes_by_class(count: int, seed: int) -> dict[str, list[int]]:
    """
    Samples graph sizes for the ground-truth experiment.

    Args:
        count (int): Number of graph sizes to sample per size class.
        seed (int): Random seed.

    Returns:
        dict[str, list[int]]: Sampled graph sizes by size class.
    """
    rng = random.Random(seed)

    return {
        size_class: [rng.randint(min_n, max_n) for _ in range(count)]
        for size_class, (min_n, max_n)
        in GROUND_TRUTH_SIZE_RANGES.items()
    }


def build_generation_tasks(output_dir: Path, sizes_by_class: dict[str, list[int]], noise_levels: list[float], base_seed: int) -> list[GroundTruthGenerationTask]:
    """
    Builds all Gaussian ground-truth graph generation tasks.

    Args:
        output_dir (Path): Root output directory.
        sizes_by_class (dict[str, list[int]]): Sampled graph sizes.
        noise_levels (list[float]): Noise levels to generate.
        base_seed (int): Base random seed.

    Returns:
        list[GroundTruthGenerationTask]: Generation tasks.
    """
    tasks: list[GroundTruthGenerationTask] = []

    configuration_index = 0

    for size_class in SIZE_CLASSES:
        sizes = sizes_by_class[size_class]

        if size_class == "large":
            community_size_classes = LARGE_COMMUNITY_SIZE_CLASSES
        else:
            community_size_classes = COMMUNITY_SIZE_CLASSES

        for regime in REGIMES:
            for community_size_class in community_size_classes:
                for noise in noise_levels:
                    configuration_seed = base_seed + 10_000_000 * configuration_index

                    for index, n in enumerate(sizes):
                        tasks.append(
                            GroundTruthGenerationTask(
                                output_dir=output_dir,
                                size_class=size_class,
                                regime=regime,
                                community_size_class=community_size_class,
                                noise=noise,
                                index=index,
                                n=n,
                                seed=configuration_seed + index * 10_000,
                            )
                        )

                    configuration_index += 1

    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gaussian partition graphs with known ground truth and explicit edge noise."
    )
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
        help="Base seed for reproducible generation.",
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

    sizes_by_class = sample_ground_truth_sizes_by_class(count=args.count, seed=args.seed)

    tasks = build_generation_tasks(
        output_dir=args.output_dir,
        sizes_by_class=sizes_by_class,
        noise_levels=NOISE_LEVELS,
        base_seed=args.seed,
    )

    print(f"Prepared {len(tasks)} ground-truth generation tasks.", flush=True)

    run_tasks(
        tasks=tasks,
        evaluate=generate_instance_task,
        workers=args.workers,
        describe=lambda task: (
            f"{task.size_class} | "
            f"{task.regime} | "
            f"communities={task.community_size_class} | "
            f"noise={task.noise:g} | "
            f"n={task.n}"
        ),
    )


if __name__ == "__main__":
    main()