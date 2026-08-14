import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from dense_graph_partition.core.graph_io import save_ground_truth_graph_json
from dense_graph_partition.core.types import Partition
from dense_graph_partition.experiments.data_generation import InstanceSpec, GenerationTask, target_average_degree, \
    generate_connected_instance, sample_sizes, sample_sizes_by_class
from dense_graph_partition.experiments.run_tasks import run_tasks


@dataclass(frozen=True)
class GroundTruthInstanceSpec(InstanceSpec):
    mu: float


@dataclass(frozen=True)
class GroundTruthGenerationTask(GenerationTask):
    mu: float


MU_VALUES = [
    0.1,
    0.3,
    0.5,
]


DEFAULT_SPECS = [
    GroundTruthInstanceSpec("partition", "sparse", "small", 0.1),
    GroundTruthInstanceSpec("partition", "sparse", "small", 0.3),
    GroundTruthInstanceSpec("partition", "sparse", "small", 0.5),

    GroundTruthInstanceSpec("partition", "dense", "small", 0.1),
    GroundTruthInstanceSpec("partition", "dense", "small", 0.3),
    GroundTruthInstanceSpec("partition", "dense", "small", 0.5),

    GroundTruthInstanceSpec("partition", "sparse", "large", 0.1),
    GroundTruthInstanceSpec("partition", "sparse", "large", 0.3),
    GroundTruthInstanceSpec("partition", "sparse", "large", 0.5),

    GroundTruthInstanceSpec("partition", "dense", "large", 0.1),
    GroundTruthInstanceSpec("partition", "dense", "large", 0.3),
    GroundTruthInstanceSpec("partition", "dense", "large", 0.5),
]


def number_of_communities(n: int, regime: str) -> int:
    """
    Determines the number of planted communities.

    Args:
        n (int): Number of nodes in the graph.
        regime (str): Density regime.

    Returns:
        int: Number of planted communities.
    """
    target_degree = target_average_degree(n, regime)

    min_community_size = math.ceil((1.0 - min(MU_VALUES)) * target_degree) + 1

    max_feasible_communities = n // min_community_size

    return max(2, min(4, max_feasible_communities),)


def balanced_community_sizes(n: int, num_communities: int) -> list[int]:
    """
    Splits the nodes into communities whose sizes differ by at most one.

    Args:
        n (int): Total number of nodes.
        num_communities (int): Number of communities.

    Returns:
        list[int]: Community sizes summing to ``n``.
    """
    base_size = n // num_communities
    remainder = n % num_communities

    return [base_size + (1 if index < remainder else 0) for index in range(num_communities)]


def partition_probabilities(n: int, sizes: list[int], regime: str, mu: float) -> tuple[float, float]:
    """
    Computes intra- and inter-community edge probabilities.

    Args:
        n (int): Number of nodes.
        sizes (list[int]): Sizes of the planted communities.
        regime (str): Density regime.
        mu (float): Expected fraction of inter-community edges.

    Returns:
        tuple[float, float]: Intra-community and inter-community edge probabilities.
    """
    target_degree = target_average_degree(n, regime)

    expected_edges = n * target_degree / 2.0

    expected_internal_edges = (1.0 - mu) * expected_edges
    expected_external_edges = mu * expected_edges

    possible_internal_edges = sum(size * (size - 1) / 2 for size in sizes)

    possible_edges = n * (n - 1) / 2
    possible_external_edges = (possible_edges - possible_internal_edges)

    p_in = expected_internal_edges / possible_internal_edges
    p_out = expected_external_edges / possible_external_edges

    if not 0.0 <= p_in <= 1.0:
        raise ValueError(f"Infeasible intra-community probability: n={n}, regime={regime}, mu={mu}, p_in={p_in:.4f}")

    if not 0.0 <= p_out <= 1.0:
        raise ValueError(f"Infeasible inter-community probability: n={n}, regime={regime}, mu={mu}, p_out={p_out:.4f}")

    return p_in, p_out


def generate_partition_graph(n: int, seed: int, regime: str, mu: float) -> nx.Graph:
    """
    Generates a random partition graph with planted communities.

    Args:
        n (int): Number of nodes.
        seed (int): Random seed used for reproducibility.
        regime (str): Density regime.
        mu (float): Expected fraction of edges between communities.

    Returns:
        nx.Graph: Generated random partition graph.
    """
    num_communities = number_of_communities(n=n, regime=regime)

    sizes = balanced_community_sizes(n=n, num_communities=num_communities)

    p_in, p_out = partition_probabilities(n=n, sizes=sizes, regime=regime, mu=mu)

    return nx.random_partition_graph(sizes=sizes, p_in=p_in, p_out=p_out, seed=seed)


def extract_ground_truth(graph: nx.Graph) -> Partition:
    """
    Extracts the planted community partition from a random partition graph.

    Args:
        graph (nx.Graph): Random partition graph.

    Returns:
        Partition: Ground-truth partition.
    """
    partition = graph.graph["partition"]

    ground_truth = [set(community) for community in partition]
    ground_truth.sort(key=lambda community: (min(community), len(community)))

    return ground_truth


def generate_instance_task(task: GroundTruthGenerationTask) -> str:
    """
    Generates and saves one connected random partition graph instance.

    Args:
        task (GroundTruthGenerationTask): Description of the graph instance to generate.

    Returns:
        str: Name of the generated instance.
    """
    graph = generate_connected_instance(generator=generate_partition_graph, n=task.n, seed=task.seed, regime=task.regime, mu=task.mu)

    ground_truth = extract_ground_truth(graph)

    mu_label = str(task.mu).replace("0.", "")

    name = (
        f"{task.size_class}_{task.graph_type}_{task.regime}_mu{mu_label}_{task.index:03d}_n{task.n}"
    )

    target_dir = task.output_dir / task.size_class / task.graph_type / task.regime / f"mu_{task.mu}"

    save_ground_truth_graph_json(G=graph, path=target_dir / f"{name}.json", name=name, ground_truth=ground_truth, mu=task.mu)

    return name


def build_generation_tasks(output_dir: Path, specs: list[GroundTruthInstanceSpec], sizes_by_class: dict[str, list[int]], base_seed: int) -> list[GroundTruthGenerationTask]:
    """
    Builds all random partition graph-generation tasks.

    Args:
        output_dir (Path): Root directory for generated graph instances.
        specs (list[GroundTruthInstanceSpec]): Instance specifications.
        sizes_by_class (dict[str, list[int]]): Graph sizes by size class.
        base_seed (int): Base seed used for reproducible generation.

    Returns:
        list[GroundTruthGenerationTask]: Graph-generation tasks.
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
                    mu=spec.mu,
                    index=index,
                    n=n,
                    seed=spec_seed + index * 10_000,
                )
            )

    return tasks


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate random partition benchmark instances with ground-truth communities for Dense Graph Partition experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ground_truth"),
        help="Directory where generated instances are stored.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of instances generated for each parameter combination.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base seed for reproducible instance generation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of worker processes. Use 1 to disable parallel execution.",
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

    print(f"Prepared {len(tasks)} random partition graph-generation tasks.", flush=True)

    run_tasks(
        tasks=tasks,
        evaluate=generate_instance_task,
        workers=args.workers,
        describe=lambda task: f"{task.graph_type} | {task.regime} | {task.size_class} | mu={task.mu} | n={task.n}",
    )


if __name__ == "__main__":
    main()