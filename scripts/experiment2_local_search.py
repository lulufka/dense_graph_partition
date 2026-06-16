import argparse
from pathlib import Path

import pandas as pd

from dense_graph_partition.experiments.algorithm_registry import START_PARTITION_NAMES
from dense_graph_partition.experiments.local_search_runner import build_phase_experiments, \
    build_local_search_tasks, run_local_search_tasks, write_local_search_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local search experiments for Dense Graph Partition."
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
        default=Path("results/experiment2"),
        help="Directory where result CSV files are written.",
    )
    parser.add_argument(
        "--phase",
        choices=["single", "short", "long", "all"],
        default="all",
        help="Pipeline phase to run.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of random runs per instance and experiment.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for reproducible runs.",
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

    experiments = build_phase_experiments(args.phase, START_PARTITION_NAMES)

    tasks = build_local_search_tasks(args.data_root, experiments, args.runs, args.base_seed)

    print(f"Prepared {len(tasks)} local-search tasks.")

    raw_rows, step_rows = run_local_search_tasks(tasks, args.workers)

    write_local_search_results(pd.DataFrame(raw_rows), pd.DataFrame(step_rows), args.results_dir)


if __name__ == "__main__":
    main()
