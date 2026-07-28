import argparse
import json
import os
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


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


DEFAULT_SPECS = [
    InstanceSpec("powerlaw", "sparse", "small"),
    InstanceSpec("powerlaw", "dense", "small"),
    InstanceSpec("er", "sparse", "small"),
    InstanceSpec("er", "dense", "small"),

    InstanceSpec("powerlaw", "sparse", "large"),
    InstanceSpec("powerlaw", "dense", "large"),
    InstanceSpec("er", "sparse", "large"),
    InstanceSpec("er", "dense", "large"),
]


def target_average_degree(n: int, regime: str) -> float:
    """
    Returns the target average node degree for a graph instance.

    Args:
        n (int): Number of nodes in the graph.
        regime (str): Density regime. Supported values are ``"sparse"`` and ``"dense"``.

    Returns:
        float: Target average node degree.

    Raises:
        ValueError: If an unknown density regime is provided.
    """
    if regime == "sparse":
        return 8.0

    if regime == "dense":
        return max(16.0, 0.04 * n)

    raise ValueError(f"Unknown regime: {regime}")


def generate_powerlaw_graph(n: int, seed: int, regime: str) -> nx.Graph:
    """
    Generates a powerlaw-cluster graph.
    Each newly added node contributes m edges, so the resulting average degree is approximately 2 * m.

    Args:
        n (int): Number of nodes.
        seed (int): Random seed used for reproducibility.
        regime (str): Density regime of the generated graph. Supported values are ``"sparse"`` and ``"dense"``.

    Returns:
        nx.Graph: The generated graph.
    """
    target_degree = target_average_degree(n, regime)

    m = max(2, round(target_degree / 2))
    m = min(m, n - 1)

    triangle_probability = 0.3

    graph = nx.powerlaw_cluster_graph(n, m, triangle_probability, seed)
    return graph


def generate_er_graph(n: int, seed: int, regime: str) -> nx.Graph:
    """
    Generates an Erdős-Rényi graph as a null model.

    Args:
        n (int): Number of nodes.
        seed (int): Random seed used for reproducibility.
        regime (str): Density regime of the generated graph. Supported values are ``"sparse"`` and ``"dense"``.

    Returns:
        nx.Graph: The generated graph.
    """
    target_degree = target_average_degree(n, regime)

    p = min(1.0, target_degree / (n - 1))

    graph = nx.erdos_renyi_graph(n, p, seed)
    return graph

GraphGenerator = Callable[[int, int, str], nx.Graph]

GENERATORS: dict[str, GraphGenerator] = {
    "powerlaw": generate_powerlaw_graph,
    "er": generate_er_graph,
}


def save_instance(path: Path, name: str, graph: nx.Graph) -> None:
    """
    Saves one graph instance as JSON.
    """
    data = {
        "name": name,
        "n": graph.number_of_nodes(),
        "m": graph.number_of_edges(),
        "edges": [list(edge) for edge in graph.edges()],
    }
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def generate_connected_instance(generator: GraphGenerator, n: int, seed: int, regime: str, max_attempts: int = 1000) -> nx.Graph:
    """
    Generates a connected graph instance.
    The graph is regenerated with consecutive seeds until a connected instance is found.

    Args:
        generator: Graph generator function.
        n: Number of nodes.
        seed: Initial random seed.
        regime: Density regime.
        max_attempts: Maximum number of generation attempts.

    Returns:
        The connected graph.

    Raises:
        RuntimeError: If no connected graph is generated within the maximum number of attempts.
    """
    for attempt in range(max_attempts):
        attempt_seed = seed + attempt
        graph = generator(n, attempt_seed, regime)

        if nx.is_connected(graph):
            return graph

    raise RuntimeError(
        f"Could not generate a connected graph after {max_attempts} attempts: n={n}, regime={regime}, initial_seed={seed}"
    )


def sample_sizes(count: int, n_min: int, n_max: int, seed: int) -> list[int]:
    rng = random.Random(seed)

    sizes = [rng.randint(n_min, n_max) for _ in range(count)]

    sizes.sort()
    return sizes


def generate_instance_task(task: GenerationTask) -> str:
    """
    Generates and saves one connected graph instance.

    Args:
        task: Description of the graph instance to generate.

    Returns:
        The generated instance name.
    """
    generator = GENERATORS[task.graph_type]

    graph = generate_connected_instance(generator=generator, n=task.n, seed=task.seed, regime=task.regime)

    name = f"{task.size_class}_{task.graph_type}_{task.regime}_{task.index:03d}_n{task.n}"

    target_dir = task.output_dir / task.size_class / task.graph_type / task.regime

    save_instance(target_dir / f"{name}.json", name, graph)

    return name


def build_generation_tasks(output_dir: Path, specs: list[InstanceSpec], small_sizes: list[int], large_sizes: list[int], base_seed: int) -> list[GenerationTask]:
    """
    Builds all graph-generation tasks.
    """
    tasks: list[GenerationTask] = []

    for spec_index, spec in enumerate(specs):
        sizes = small_sizes if spec.size_class == "small" else large_sizes

        spec_seed = base_seed + 10_000_000 * spec_index

        for index, n in enumerate(sizes):
            tasks.append(
                GenerationTask(
                    output_dir=output_dir,
                    graph_type=spec.graph_type,
                    regime=spec.regime,
                    size_class=spec.size_class,
                    index=index,
                    n=n,
                    seed=spec_seed + index * 10_000,
                )
            )

    return tasks


def run_generation_tasks(tasks: list[GenerationTask], workers: int) -> None:
    """
    Generates and saves all graph instances.

    Args:
        tasks: Graph-generation tasks.
        workers: Number of worker processes. A value of 1 disables parallel execution.
    """
    total_tasks = len(tasks)

    if workers <= 1:
        for completed, task in enumerate(tasks, start=1):
            name = generate_instance_task(task)

            print(f"[{completed}/{total_tasks}] {name}", flush=True)

        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate_instance_task, task): task for task in tasks}

        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]

            try:
                name = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"Graph generation failed: graph_type={task.graph_type}, regime={task.regime}, size_class={task.size_class}, index={task.index}, n={task.n}"
                ) from error

            print(f"[{completed}/{total_tasks}] {name}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic graph instances for Dense Graph Partition experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory where generated instances are stored.",
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
        default=max(1, (os.cpu_count() or 1) - 1),
        help=("Number of worker processes. Use 1 to disable parallel execution."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    small_sizes = sample_sizes(count=250, n_min=50, n_max=250, seed=args.seed)
    large_sizes = sample_sizes(count=250, n_min=500, n_max=1500, seed=args.seed + 1)

    tasks = build_generation_tasks(
        output_dir=args.output_dir,
        specs=DEFAULT_SPECS,
        small_sizes=small_sizes,
        large_sizes=large_sizes,
        base_seed=args.seed,
    )

    print(f"Prepared {len(tasks)} graph-generation tasks.", flush=True)

    run_generation_tasks(tasks=tasks, workers=args.workers)


if __name__ == "__main__":
    main()


