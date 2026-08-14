import argparse
from pathlib import Path

import networkx as nx

from dense_graph_partition.core.graph_io import save_graph_json
from dense_graph_partition.experiments.data_generation import InstanceSpec, target_average_degree, GraphGenerator, \
    GenerationTask, generate_connected_instance, sample_sizes, sample_sizes_by_class
from dense_graph_partition.experiments.run_tasks import run_tasks

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

    return nx.powerlaw_cluster_graph(n, m, triangle_probability, seed)


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

    return nx.erdos_renyi_graph(n, p, seed)


GENERATORS: dict[str, GraphGenerator] = {
    "powerlaw": generate_powerlaw_graph,
    "er": generate_er_graph,
}


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

    save_graph_json(G=graph, path=target_dir / f"{name}.json", name=name)

    return name


def build_generation_tasks(output_dir: Path, specs: list[InstanceSpec], sizes_by_class: dict[str, list[int]], base_seed: int) -> list[GenerationTask]:
    """
    Builds all graph-generation tasks.

    Args:
        output_dir (Path): Root directory for generated instances.
        specs (list[InstanceSpec]): Instance specifications.
        sizes_by_class (dict[str, list[int]]): Graph sizes by size class.
        base_seed (int): Base seed for reproducibility.

    Returns:
        list[GenerationTask]: Graph-generation tasks.
    """
    tasks: list[GenerationTask] = []

    for spec_index, spec in enumerate(specs):
        sizes = sizes_by_class[spec.size_class]

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
        default=50,
        help=("Number of worker processes. Use 1 to disable parallel execution."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sizes_by_class = sample_sizes_by_class(count=250, seed=args.seed,)

    tasks = build_generation_tasks(
        output_dir=args.output_dir,
        specs=DEFAULT_SPECS,
        sizes_by_class=sizes_by_class,
        base_seed=args.seed,
    )

    print(f"Prepared {len(tasks)} graph-generation tasks.", flush=True)

    run_tasks(
        tasks=tasks,
        evaluate=generate_instance_task,
        workers=args.workers,
        describe=lambda task: f"{task.graph_type} | {task.regime} | {task.size_class} | n={task.n}",
    )


if __name__ == "__main__":
    main()


