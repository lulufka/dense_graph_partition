import argparse
import json
import random
from pathlib import Path

import networkx as nx
from dataclasses import dataclass


@dataclass(frozen=True)
class InstanceSpec:
    graph_type: str
    regime: str
    size_class: str
    count: int
    n_min: int
    n_max: int


DEFAULT_SPECS = [
    InstanceSpec("partition", "sparse", "small", 40, 50, 250),
    InstanceSpec("partition", "dense", "small", 40, 50, 250),
    InstanceSpec("powerlaw", "sparse", "small", 40, 50, 250),
    InstanceSpec("powerlaw", "dense", "small", 40, 50, 250),
    InstanceSpec("er", "sparse", "small", 40, 50, 250),
    InstanceSpec("er", "dense", "small", 40, 50, 250),

    InstanceSpec("partition", "sparse", "large", 20, 500, 1500),
    InstanceSpec("partition", "dense", "large", 20, 500, 1500),
    InstanceSpec("powerlaw", "sparse", "large", 20, 500, 1500),
    InstanceSpec("powerlaw", "dense", "large", 20, 500, 1500),
    InstanceSpec("er", "sparse", "large", 20, 500, 1500),
    InstanceSpec("er", "dense", "large", 20, 500, 1500),
]


def balanced_random_partition_sizes(n: int, k: int, rng: random.Random, min_size: int = 4) -> list[int]:
    """
    Generates k community sizes that sum to n.

    Args:
        n (int): Total number of nodes.
        k (int): Number of communities.
        rng (random.Random): Random number generator used for reproducibility.
        min_size (int): Minimum community size. Defaults to 4.

    Returns:
        list[int]: A list of ``k`` positive integers whose sum is ``n``. Every value is at least ``min_size``.
    """
    if n < k * min_size:
        raise ValueError("Too many communities for the requested minimum size.")

    sizes = [min_size] * k
    remaining = n - k * min_size

    for _ in range(remaining):
        sizes[rng.randrange(k)] += 1

    rng.shuffle(sizes)
    return sizes


def sample_num_communities(n: int, rng: random.Random, min_size: int=4) -> int:
    """
    Samples the number of planted communities based on graph size.
    """
    min_avg_size = max(8, round(n**0.4))
    max_avg_size = max(25, round(n**0.55))

    k_min = max(2, round(n / max_avg_size))
    k_max = max(k_min, round(n / min_avg_size))
    k_max = min(k_max, n // min_size)

    return rng.randint(k_min, k_max)


def generate_partition_graph(n: int, seed: int, regime: str) -> tuple[nx.Graph, list[list[int]]]:
    """
    Generates a graph with planted communities. Communities are generated explicitly and serve as a ground truth partition.
    Internal edge probabilities are chosen significantly higher than external edge probabilities to create a detectable community structure.

    Args:
        n (int): Number of nodes
        seed (int): Random seed used for reproducibility.
        regime (str): Density regime of the generated graph. Supported values are ``"sparse"`` and ``"dense"``.

    Returns:
        tuple[nx.Graph, list[list[int]]]: The generated graph and the planted communities.
    """
    rng = random.Random(seed)
    k = sample_num_communities(n, rng)
    sizes = balanced_random_partition_sizes(n, k, rng)

    average_community_size = n / k

    if regime == "sparse":
        target_internal_degree = 1.6 * (n**0.25)
        target_external_degree = 0.2 * (n**0.25)
    elif regime == "dense":
        target_internal_degree = max(8.0, 0.015 * n)
        target_external_degree = max(2.0, 0.004 * n)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    p_in = min(1.0, target_internal_degree / (average_community_size - 1))
    p_out = min(1.0, target_external_degree / (n - average_community_size))

    graph = nx.random_partition_graph(sizes, p_in, p_out, seed)
    communities = [sorted(cluster) for cluster in graph.graph["partition"]]
    return graph, communities


def generate_powerlaw_graph(n: int, seed: int, regime: str) -> tuple[nx.Graph, list[list[int]]]:
    """
    Generates a powerlaw-cluster graph.

    Args:
        n (int): Number of nodes.
        seed (int): Random seed used for reproducibility.
        regime (str): Density regime of the generated graph. Supported values are ``"sparse"`` and ``"dense"``.

    Returns:
        tuple[nx.Graph, list[list[int]]]: The generated graph and an empty community list.
    """
    if regime == "sparse":
        m = max(2, round(0.9 * (n**0.25)))
        triangle_probability = 0.15
    elif regime == "dense":
        m = max(4, round(0.01 * n))
        triangle_probability = 0.30
    else:
        raise ValueError(f"Unknown regime: {regime}")

    m = min(m, n - 1)

    graph = nx.powerlaw_cluster_graph(n, m, triangle_probability, seed)
    return graph, []


def generate_er_graph(n: int, seed: int, regime: str) -> tuple[nx.Graph, list[list[int]]]:
    """
    Generates an Erdős-Rényi graph as a null model without planted communities.

    Args:
        n (int): Number of nodes.
        seed (int): Random seed used for reproducibility.
        regime (str): Density regime of the generated graph. Supported values are ``"sparse"`` and ``"dense"``.

    Returns:
        tuple[nx.Graph, list[list[int]]]: The generated graph and an empty community list.
    """
    if regime == "sparse":
        target_degree = max(6.0, 1.8 * (n**0.25))
    elif regime == "dense":
        target_degree = max(12.0, 0.04 * n)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    p = min(1.0, target_degree / (n - 1))

    graph = nx.erdos_renyi_graph(n, p, seed)
    return graph, []



GENERATORS = {
    "partition": generate_partition_graph,
    "powerlaw": generate_powerlaw_graph,
    "er": generate_er_graph,
}


def save_instance(path: Path, name: str, graph: nx.Graph, communities: list[list[int]]) -> None:
    """
    Saves one graph instance as JSON.
    """
    data = {
        "name": name,
        "n": graph.number_of_nodes(),
        "m": graph.number_of_edges(),
        "edges": [list(edge) for edge in graph.edges()],
        "communities": communities
    }
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def generate_spec(output_dir: Path, spec: InstanceSpec, base_seed: int) -> None:
    """
    Generates and saves all instances for one instance specification.
    """
    rng = random.Random(base_seed)
    generator = GENERATORS[spec.graph_type]
    instances = []

    for index in range(spec.count):
        n = rng.randint(spec.n_min, spec.n_max)
        seed = base_seed + index
        graph, communities = generator(n, seed, spec.regime)
        instances.append((n, index, seed, graph, communities))

    instances.sort(key=lambda item: (item[0], item[1]))

    target_dir = output_dir / spec.size_class / spec.graph_type / spec.regime

    for sorted_index, (n, _, seed, graph, communities) in enumerate(instances):
        name = (f"{spec.graph_type}_{spec.regime}_{spec.size_class}_{sorted_index:03d}_n{n}_s{seed}")
        save_instance(target_dir / f"{name}.json", name, graph, communities)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic graph instances for Dense Graph Partition experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/generated"),
        help="Directory where generated instances are stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base seed for reproducible instance generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for spec_index, spec in enumerate(DEFAULT_SPECS):
        generate_spec(
            output_dir=args.output_dir,
            spec=spec,
            base_seed=args.seed + 10_000 * spec_index,
        )


if __name__ == "__main__":
    main()


