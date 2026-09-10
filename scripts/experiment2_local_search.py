import argparse
from pathlib import Path

import pandas as pd

from dense_graph_partition.experiments.algorithm_registry import STARTPARTITIONS
from dense_graph_partition.experiments.extra_operator_runner import build_branching_local_search_tasks, build_extra_operator_experiments, \
    run_branching_local_search_tasks, write_branching_local_search_results
from dense_graph_partition.experiments.local_search_runner import build_local_search_experiments, build_local_search_tasks, \
    run_local_search_tasks, write_local_search_results


EXPERIMENTS = (
    "move_strategies",
    "repeated_plateau_search",
    "zero_gain_limit",
    "initial_partition",
    "additional_local_search_operator",
    "independent_runs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local-search experiments for Dense Graph Partition."
    )
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS,
        required=True,
        help="Local-search experiment to run.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/generated/small"),
        help="Root directory containing graph instances.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory where result CSV files are written.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of randomized runs. Uses the experiment default if omitted.",
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


def run_local_search_experiment(
        args: argparse.Namespace,
        start_partitions: list[str],
        plateau_steps: int,
        zero_gain_factors: tuple[int, ...],
        include_move_operators: bool,
        runs: int,
        results_dir: Path,
) -> None:
    experiments = build_local_search_experiments(
        start_partitions=start_partitions,
        plateau_steps=plateau_steps,
        zero_gain_factors=zero_gain_factors,
        include_move_operators=include_move_operators,
    )

    tasks = build_local_search_tasks(args.data_root, experiments, runs, args.base_seed)
    print(f"Prepared {len(tasks)} local-search tasks.")
    raw_rows, step_rows = run_local_search_tasks(tasks, args.workers)
    write_local_search_results(pd.DataFrame(raw_rows), pd.DataFrame(step_rows), results_dir)


def main() -> None:
    args = parse_args()

    default_results_dir = Path("results/experiment2") / args.experiment
    results_dir = args.results_dir or default_results_dir

    if args.experiment == "move_strategies":
        run_local_search_experiment(
            args=args,
            start_partitions=STARTPARTITIONS,
            plateau_steps=1,
            zero_gain_factors=(2,),
            include_move_operators=True,
            runs=args.runs or 10,
            results_dir=results_dir,
        )

    elif args.experiment == "repeated_plateau_search":
        run_local_search_experiment(
            args=args,
            start_partitions=STARTPARTITIONS,
            plateau_steps=10,
            zero_gain_factors=(2,),
            include_move_operators=False,
            runs=args.runs or 10,
            results_dir=results_dir,
        )

    elif args.experiment == "zero_gain_limit":
        run_local_search_experiment(
            args=args,
            start_partitions=STARTPARTITIONS,
            plateau_steps=4,
            zero_gain_factors=(1, 2, 4, 8),
            include_move_operators=False,
            runs=args.runs or 10,
            results_dir=results_dir,
        )

    elif args.experiment == "initial_partition":
        run_local_search_experiment(
            args=args,
            start_partitions=STARTPARTITIONS,
            plateau_steps=4,
            zero_gain_factors=(4,),
            include_move_operators=False,
            runs=args.runs or 10,
            results_dir=results_dir,
        )

    elif args.experiment == "additional_local_search_operator":
        runs = args.runs or 10

        experiments = build_extra_operator_experiments(
            start_partitions=["maximum_matching"],
            plateau_steps=4,
            zero_gain_factor=4,
        )

        tasks = build_branching_local_search_tasks(args.data_root, experiments, runs, args.base_seed)
        print(f"Prepared {len(tasks)} branching-local-search tasks.")
        raw_rows, step_rows = run_branching_local_search_tasks(tasks, args.workers,)
        write_branching_local_search_results(pd.DataFrame(raw_rows), pd.DataFrame(step_rows), results_dir)

    elif args.experiment == "independent_runs":
        run_local_search_experiment(
            args=args,
            start_partitions=["maximum_matching"],
            plateau_steps=4,
            zero_gain_factors=(4,),
            include_move_operators=False,
            runs=args.runs or 50,
            results_dir=results_dir,
        )


if __name__ == "__main__":
    main()