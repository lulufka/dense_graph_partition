import argparse
from pathlib import Path

import pandas as pd

from dense_graph_partition.experiments.algorithm_registry import build_algorithm_specs, STARTPARTITIONS
from dense_graph_partition.experiments.runner import build_algorithm_tasks, run_algorithm_tasks, write_raw_results


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
        default=50,
        help="Number of parallel worker processes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    algorithms = build_algorithm_specs(STARTPARTITIONS)

    tasks = build_algorithm_tasks(data_root=args.data_root, algorithms=algorithms)

    print(f"Prepared {len(tasks)} baseline tasks.")

    raw_rows = run_algorithm_tasks(tasks=tasks, workers=args.workers)

    write_raw_results(raw_results=pd.DataFrame(raw_rows), results_dir=args.results_dir)



if __name__ == "__main__":
    main()