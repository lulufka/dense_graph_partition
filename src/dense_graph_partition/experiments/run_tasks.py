from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_tasks(tasks: list[T], evaluate: Callable[[T], R], workers: int, describe: Callable[[T], str]) -> list[R]:
    """
    Runs experiment tasks sequentially or in parallel.

    Args:
        tasks (list[T]): Tasks to evaluate.
        evaluate (Callable[[T], R]): Function used to evaluate one task.
        workers (int): Number of worker processes.
        describe (Callable[[T], str]): Function used to describe a task in progress output.

    Returns:
        list[R]: Evaluation results in completion order.
    """
    results: list[R] = []

    if workers <= 1:
        for index, task in enumerate(tasks):
            results.append(evaluate(task))

            print(f"[{index + 1}/{len(tasks)}] {describe(task)}")

        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate, task): task
            for task in tasks
        }

        for index, future in enumerate(as_completed(futures)):
            task = futures[future]

            results.append(future.result())

            print(f"[{index + 1}/{len(tasks)}] {describe(task)}")

    return results