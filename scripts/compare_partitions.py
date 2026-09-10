import argparse
import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

import cairosvg
import networkx as nx
import pandas as pd

from dense_graph_partition.core.evaluation import cluster_density, partition_density
from dense_graph_partition.core.types import Partition


@dataclass(frozen=True)
class SvgStyle:
    width: int = 2000
    min_height: int = 700
    margin: int = 28
    title_height: int = 86
    cluster_cell_width: int = 240
    cluster_cell_height: int = 180
    node_radius: float = 9.0
    edge_width: float = 1.8


@dataclass(frozen=True)
class Panel:
    title: str
    partition: Partition
    is_best: bool = False


@dataclass(frozen=True)
class ClusterStructure:
    cluster: set[int]
    count: int


ALGORITHMS = {
    "leiden": "Leiden",
    "leiden_mdgp": "Leiden-MDGP",
    "kapoce": "KaPoCE",
    "mdgp_plateau": "MDGP-Plateau",
}


def format_page_title(instance_name: str, G: nx.Graph) -> str:
    size_class = "large" if instance_name.startswith("large") else "small"
    density_class = "dense" if "dense" in instance_name else "sparse"
    community_size = "large" if "communities-large" in instance_name else "small"

    match = re.search(r"noise-([0-9]+(?:-[0-9]+)?)", instance_name)
    noise = match.group(1).replace("-", ".") if match else "0.05"

    return (
        f"Dataset: {size_class} {density_class} | "
        f"Noise: {noise} | "
        f"Community size: {community_size} | "
        f"Vertices: {G.number_of_nodes()} | "
        f"Edges: {G.number_of_edges()}"
    )


def load_instance(path: Path) -> tuple[str, nx.Graph, Partition]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    G = nx.Graph()
    G.add_nodes_from(range(data["n"]))
    G.add_edges_from((int(u), int(v)) for u, v in data["edges"])

    ground_truth = [set(cluster) for cluster in data["ground_truth"]]
    return data["name"], G, ground_truth


def load_algorithm_partitions(results: pd.DataFrame, instance_name: str) -> dict[str, Partition]:
    return {
        algorithm: [
            set(cluster)
            for cluster in json.loads(
                results.loc[(results["instance"] == instance_name) & (results["algorithm"] == algorithm), "partition"].iloc[0]
            )
        ]
        for algorithm in ALGORITHMS
    }


def induced_graph(G: nx.Graph, cluster: set[int]) -> nx.Graph:
    return G.subgraph(cluster).copy()


def visible_clusters(G: nx.Graph, partition: Partition) -> list[set[int]]:
    return sorted(
        (set(cluster) for cluster in partition),
        key=lambda cluster: (
            -len(cluster),
            -G.subgraph(cluster).number_of_edges(),
            sorted(cluster),
        ),
    )


def group_isomorphic_clusters(G: nx.Graph, partition: Partition) -> list[ClusterStructure]:
    groups: list[ClusterStructure] = []

    for cluster in visible_clusters(G, partition):
        cluster_graph = induced_graph(G, cluster)

        for i, group in enumerate(groups):
            representative_graph = induced_graph(G, group.cluster)

            if (
                    cluster_graph.number_of_nodes() == representative_graph.number_of_nodes()
                    and cluster_graph.number_of_edges() == representative_graph.number_of_edges()
                    and nx.is_isomorphic(cluster_graph, representative_graph)
            ):
                groups[i] = ClusterStructure(group.cluster, group.count + 1)
                break
        else:
            groups.append(ClusterStructure(cluster, 1))

    return groups


def panel_columns(panel_width: float, style: SvgStyle) -> int:
    cell_gap = 18
    return max(1, int((panel_width + cell_gap) // (style.cluster_cell_width + cell_gap)))


def page_height(G: nx.Graph, panels: list[Panel], style: SvgStyle) -> int:
    panel_count = len(panels)
    panel_width = (style.width - 2 * style.margin - style.margin * (panel_count - 1)) / panel_count
    columns = panel_columns(panel_width, style)

    max_rows = max(
        max(1, (len(group_isomorphic_clusters(G, panel.partition)) + columns - 1) // columns)
        for panel in panels
    )

    return max(style.min_height, style.title_height + max_rows * style.cluster_cell_height + style.margin)


def cluster_layout_seed(cluster: set[int], seed: int) -> int:
    return seed + sum((i + 1) * node for i, node in enumerate(sorted(cluster)))


def scale_positions(
        positions: dict[int, tuple[float, float]],
        x: float,
        y: float,
        width: float,
        height: float,
) -> dict[int, tuple[float, float]]:
    if len(positions) == 1:
        node = next(iter(positions))
        return {node: (x + width / 2, y + height / 2)}

    xs = [position[0] for position in positions.values()]
    ys = [position[1] for position in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    padding = max(min(width, height) * 0.08, 0.0)
    drawable_width = max(width - 2 * padding, 1.0)
    drawable_height = max(height - 2 * padding, 1.0)

    return {
        node: (
            x + padding + ((position[0] - min_x) / span_x) * drawable_width,
            y + padding + ((position[1] - min_y) / span_y) * drawable_height,
        )
        for node, position in positions.items()
    }


def cluster_layout(
        G: nx.Graph,
        x: float,
        y: float,
        width: float,
        height: float,
        seed: int,
) -> dict[int, tuple[float, float]]:
    return scale_positions(nx.spring_layout(G, seed=seed), x, y, width, height)


def summary_text(G: nx.Graph, partition: Partition, structure_count: int, is_best: bool) -> str:
    density = partition_density(G, partition)
    density_text = f'<tspan font-weight="700">{density:.3f}</tspan>' if is_best else f"{density:.3f}"

    return (
        f"Clusters: {len(partition)} | "
        f"Structures: {structure_count} | "
        f"Density: {density_text}"
    )


def render_cluster_cell(
        G: nx.Graph,
        structure: ClusterStructure,
        display_index: int,
        x: float,
        y: float,
        style: SvgStyle,
        seed: int,
        show_labels: bool,
) -> list[str]:
    cluster = structure.cluster
    subgraph = induced_graph(G, cluster)

    title = (
        f"#{display_index}: "
        f"|V|={len(cluster)}, "
        f"|E|={subgraph.number_of_edges()}, "
        f"ρ(C)={cluster_density(G, cluster):.2f}, "
        f"Count={structure.count}"
    )

    graph_x = x + 18
    graph_y = y + 34
    graph_width = style.cluster_cell_width - 36
    graph_height = style.cluster_cell_height - 52

    positions = cluster_layout(
        subgraph,
        graph_x,
        graph_y,
        graph_width,
        graph_height,
        cluster_layout_seed(cluster, seed),
    )

    lines = [
        f'<text class="label" x="{x:.2f}" y="{y + 14:.2f}">{escape(title)}</text>'
    ]

    for u, v in subgraph.edges():
        x1, y1 = positions[int(u)]
        x2, y2 = positions[int(v)]
        lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="#627d98" stroke-opacity="0.72" '
            f'stroke-width="{style.edge_width}" />'
        )

    for node in sorted(subgraph.nodes()):
        node_x, node_y = positions[int(node)]

        lines.append(
            f'<circle cx="{node_x:.2f}" cy="{node_y:.2f}" '
            f'r="{style.node_radius}" fill="#334e68" '
            f'stroke="#ffffff" stroke-width="1.6">'
            f"<title>Vertex {escape(str(node))}</title></circle>"
        )

        if show_labels:
            lines.append(
                f'<text class="node-label" x="{node_x:.2f}" y="{node_y:.2f}">{escape(str(node))}</text>'
            )

    return lines


def render_panel(
        G: nx.Graph,
        panel: Panel,
        x: float,
        y: float,
        width: float,
        style: SvgStyle,
        seed: int,
        show_labels: bool,
) -> list[str]:
    structures = group_isomorphic_clusters(G, panel.partition)
    columns = panel_columns(width, style)
    cell_gap = 18

    lines = [
        f'<text class="panel-title" x="{x:.2f}" y="{y - 36:.2f}">{escape(panel.title)}</text>',
        f'<text class="subtitle" x="{x:.2f}" y="{y - 16:.2f}">{summary_text(G, panel.partition, len(structures), panel.is_best)}</text>',
    ]

    for i, structure in enumerate(structures):
        cell_x = x + (i % columns) * (style.cluster_cell_width + cell_gap)
        cell_y = y + (i // columns) * style.cluster_cell_height
        lines.extend(render_cluster_cell(G, structure, i + 1, cell_x, cell_y, style, seed, show_labels))

    return lines


def render_svg(
        G: nx.Graph,
        panels: list[Panel],
        title: str,
        style: SvgStyle,
        seed: int,
        show_labels: bool,
) -> str:
    height = page_height(G, panels, style)
    panel_count = len(panels)
    gap = style.margin
    panel_width = (style.width - 2 * style.margin - gap * (panel_count - 1)) / panel_count

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{style.width}" height="{height}" viewBox="0 0 {style.width} {height}">',
        "<style>",
        ".title { font: 700 20px Arial, sans-serif; fill: #1f2933; }",
        ".panel-title { font: 700 15px Arial, sans-serif; fill: #1f2933; }",
        ".subtitle { font: 13px Arial, sans-serif; fill: #52606d; }",
        ".label { font: 12px Arial, sans-serif; fill: #323f4b; }",
        ".node-label { font: 11px Arial, sans-serif; fill: #ffffff; text-anchor: middle; dominant-baseline: central; pointer-events: none; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text class="title" x="{style.margin}" y="34">{escape(title)}</text>',
    ]

    for i, panel in enumerate(panels):
        x = style.margin + i * (panel_width + gap)
        lines.extend(render_panel(G, panel, x, style.title_height, panel_width, style, seed, show_labels))

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_partition_comparison_svg(
        G: nx.Graph,
        algorithm_partitions: dict[str, Partition],
        ground_truth: Partition,
        output_path: Path,
        title: str,
        seed: int,
        show_labels: bool,
        save_png: bool,
) -> None:
    densities = {
        algorithm: partition_density(G, partition)
        for algorithm, partition in algorithm_partitions.items()
    }
    best_density = max(densities.values())

    panels = [Panel("Ground Truth", ground_truth)]

    for algorithm, display_name in ALGORITHMS.items():
        panels.append(
            Panel(
                display_name,
                algorithm_partitions[algorithm],
                abs(densities[algorithm] - best_density) <= 1e-9,
                )
        )

    svg = render_svg(G, panels, title, SvgStyle(), seed, show_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")

    if save_png:
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(output_path.with_suffix(".png")),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--results", type=Path, default=Path("results/experiment4/raw_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--png", action="store_true", help="Additionally save the comparison as PNG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results)

    instance_name, G, ground_truth = load_instance(args.input)
    algorithm_partitions = load_algorithm_partitions(results, instance_name)

    output_path = args.output_dir / f"{instance_name}_comparison.svg"

    write_partition_comparison_svg(
        G,
        algorithm_partitions,
        ground_truth,
        output_path,
        format_page_title(instance_name, G),
        args.seed,
        not args.no_labels,
        args.png,
    )

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()