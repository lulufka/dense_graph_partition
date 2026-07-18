import argparse
from pathlib import Path

import pandas as pd
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(args.data_root)

    rows = []

    for i, dataset in enumerate(datasets, start=1):
        if not dataset.path.exists():
            raise FileNotFoundError(f"Dataset directory {dataset.path} does not exist.")

        print(f"[{i}/{len(datasets)}] Running {dataset.name}")

        dataset_rows = run_dataset(dataset.path, dataset.name, dataset.size_class, dataset.graph_type, dataset.regime, args.workers)

        rows.extend(dataset_rows)

        print(f"    Finished ({len(dataset_rows)} runs)")

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