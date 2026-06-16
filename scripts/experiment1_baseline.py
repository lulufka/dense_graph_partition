import argparse
from pathlib import Path

import pandas as pd
from dense_graph_partition.experiments.algorithm_registry import build_baseline_algorithm_specs
from dense_graph_partition.experiments.datasets import build_datasets
from dense_graph_partition.experiments.baseline_runner import run_dataset, add_relative_scores, summarize_results, \
    thesis_summary_table, overall_thesis_summary_table, rounded_for_export


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

    algorithms = build_baseline_algorithm_specs()

    rows = []

    for data_dir, dataset_name, size_class, graph_type, regime in build_datasets(args.data_root):
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