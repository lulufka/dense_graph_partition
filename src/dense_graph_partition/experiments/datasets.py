from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """
    Describes one generated dataset.

    Attributes:
        path (Path): Directory containing graph instances.
        name (str): Dataset name used in result tables.
        size_class (str): Size category, e.g. ``"small"`` or ``"large"``.
        graph_type (str): Graph generator type.
        regime (str): Density regime.
    """
    path: Path
    name: str
    size_class: str
    graph_type: str
    regime: str


def build_datasets(data_root: Path) -> list[DatasetSpec]:
    """
    Enumerates all generated datasets used by the experiments.

    Args:
        data_root (Path): Root directory containing generated graph instances.

    Returns:
        list[DatasetSpec]: Dataset specifications.
    """
    datasets: list[DatasetSpec] = []

    for size_class in ["small", "large"]:
        for graph_type in ["partition", "powerlaw", "er"]:
            for regime in ["sparse", "dense"]:
                path = data_root / size_class / graph_type / regime
                name = f"{graph_type}_{regime}_{size_class}"

                datasets.append(
                    DatasetSpec(
                        path=path,
                        name=name,
                        size_class=size_class,
                        graph_type=graph_type,
                        regime=regime,
                    )
                )

    return datasets