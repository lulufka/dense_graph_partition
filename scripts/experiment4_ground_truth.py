import argparse
from pathlib import Path

import pandas as pd

from dense_graph_partition.experiments.algorithm_registry import COMPARISON_ALGORITHMS, build_algorithm_specs
from dense_graph_partition.experiments.runner import write_raw_results

from dense_graph_partition.experiments.ground_truth_runner import build_ground_truth_tasks, run_ground_truth_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate partitioning algorithms on synthetic graphs with known ground truth communities."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/ground_truth"),
        help="Root directory containing ground-truth graph instances.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/experiment4"),
        help="Directory where result CSV files are written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of parallel worker processes.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    algorithms = build_algorithm_specs(COMPARISON_ALGORITHMS)

    tasks = build_ground_truth_tasks(data_root=args.data_root, algorithms=algorithms)

    print(f"Prepared {len(tasks)} ground-truth tasks.")

    raw_rows = run_ground_truth_tasks(tasks=tasks, workers=args.workers)

    write_raw_results(raw_results=pd.DataFrame(raw_rows), results_dir=args.results_dir)


if __name__ == "__main__":
    main()