import argparse
from functools import partial
from pathlib import Path

import pandas as pd
from dense_graph_partition.adapters.kapoce import kapoce_partition
from dense_graph_partition.adapters.leiden import leiden_mdgp_partition
from dense_graph_partition.algorithms.matching import matching_partition, maximum_matching_partition, \
    high_degree_first_matching_partition, high_degree_product_matching_partition
from dense_graph_partition.config import load_kapoce_config
from dense_graph_partition.experiments.runner import AlgorithmSpec, run_dataset, add_relative_scores, summarize_results, \
    thesis_summary_table, overall_thesis_summary_table, rounded_for_export


def build_algorithms() ->list[AlgorithmSpec]:
    """
    Builds the list of algorithms evaluated in the first baseline experiment.

    Returns:
        list[AlgorithmSpec]: Algorithm specifications used by the experiment runner.
    """
    kapoce_config = load_kapoce_config()

    return [
        AlgorithmSpec(
            name="matching",
            run=matching_partition
        ),
        AlgorithmSpec(
            name="maximum_matching",
            run=maximum_matching_partition
        ),
        AlgorithmSpec(
            name="high_degree_first_matching",
            run=high_degree_first_matching_partition
        ),
        AlgorithmSpec(
            name="high_degree_product_matching",
            run=high_degree_product_matching_partition
        ),
        AlgorithmSpec(
            name="leiden_mdgp",
            run=leiden_mdgp_partition
        ),
        AlgorithmSpec(
            name="kapoce",
            run=partial(kapoce_partition, executable_path=kapoce_config.executable_path, config_path=kapoce_config.config_path)
        )
    ]


def iter_datasets(data_root: Path) -> list[tuple[Path, str, str, str, str]]:
    """
    Enumerates all generated datasets used in the first baseline experiment.

    Args:
        data_root (Path): Root directory containing generated graph instances.

    Returns:
        list[tuple[Path, str, str, str, str]]: Tuples containing dataset path, dataset name, size class, graph type, and density regime.
    """
    datasets = []

    for size_class in ["small", "large"]:
        for graph_type in ["partition", "powerlaw", "er"]:
            for regime in ["sparse", "dense"]:
                data_dir = data_root / size_class / graph_type / regime
                dataset_name = f"{graph_type}_{regime}_{size_class}"

                datasets.append((data_dir, dataset_name, size_class, graph_type, regime))

    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline algorithms for Dense Graph Partition experiments."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/generated"),
        help="Root directory containing graph instances",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/experiment1"),
        help="Directory where result CSV files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    algorithms = build_algorithms()

    rows = []

    for data_dir, dataset_name, size_class, graph_type, regime in iter_datasets(args.data_root):
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory {data_dir} does not exist.")

        rows.extend(
            run_dataset(data_dir, algorithms, dataset_name, size_class, graph_type, regime)
        )

    raw_results = pd.DataFrame(rows)
    raw_results = add_relative_scores(raw_results)

    summary = summarize_results(raw_results)
    thesis_table = thesis_summary_table(summary)
    overall_table = overall_thesis_summary_table(raw_results)

    rounded_for_export(raw_results).to_csv(args.results_dir / "raw_results.csv", index=False)
    rounded_for_export(summary).to_csv(args.results_dir / "summary.csv", index=False)
    rounded_for_export(thesis_table).to_csv(args.results_dir / "thesis_summary_table.csv", index=False)
    rounded_for_export(overall_table).to_csv(args.results_dir / "overall_summary.csv", index=False)


if __name__ == "__main__":
    main()