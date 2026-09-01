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


def build_single_dataset(data_dir: Path) -> DatasetSpec:
    """
    Builds one dataset specification from a concrete instance directory.

    Args:
        data_dir (Path): Directory containing graph instances with expected path layout: <root>/<size_class>/<graph_type>/<regime>.
    """
    regime = data_dir.name
    graph_type = data_dir.parent.name
    size_class = data_dir.parent.parent.name
    name = f"{graph_type}_{regime}_{size_class}"

    return DatasetSpec(
        path=data_dir,
        name=name,
        size_class=size_class,
        graph_type=graph_type,
        regime=regime,
    )


def build_datasets(data_root: Path) -> list[DatasetSpec]:
    """
    Enumerates generated datasets used by the experiments.

    Args:
        data_root (Path): Root directory containing generated graph instances.

    Returns:
        list[DatasetSpec]: Dataset specifications.
    """
    if any(data_root.glob("*.json")):
        return [build_single_dataset(data_root)]

    datasets: list[DatasetSpec] = []

    if data_root.name in {"small", "large"}:
        size_classes = [data_root.name]
        root = data_root.parent
    else:
        size_classes = ["small", "large"]
        root = data_root

    for size_class in size_classes:
        for graph_type in ["powerlaw", "er"]:
            for regime in ["sparse", "dense"]:
                path = root / size_class / graph_type / regime
                name = f"{graph_type}_{regime}_{size_class}"

                datasets.append(
                    DatasetSpec(path=path, name=name, size_class=size_class, graph_type=graph_type, regime=regime)
                )

    return datasets