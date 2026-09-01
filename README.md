# Heuristics for Dense Graph Partition

This repository contains the implementation and experimental evaluation developed for the Master's thesis **"Heuristics for Dense Graph Partition"**.

The project considers the **Max Dense Graph Partition (MDGP)** problem. Given an undirected graph, the objective is to partition its vertices into clusters such that the sum of the internal cluster densities is maximized.

## Installation

A setup script is provided that installs the Python environment and builds the external dependencies required by the project.

### Clone the repository

```bash
git clone https://github.com/lulufka/dense_graph_partition.git
cd dense_graph_partition
```

### Run the setup

The complete development environment can be created with:

```bash
./setup.sh
```

The setup script automatically:

1. creates a Python virtual environment in `.venv`,
2. installs the Python package and development dependencies,
3. builds native `igraph` 1.0.0,
4. builds the customized `libleidenalg`,
5. installs the customized Python `leidenalg` package,
6. builds the KaPoCE Cluster Editing heuristic,
7. creates the local KaPoCE configuration, and
8. performs basic checks of the Leiden-MDGP and KaPoCE installations.

The additional source repositories are cloned next to the `dense_graph_partition` repository. The resulting directory structure is:

```text
Repositories/
├── dense_graph_partition/
├── igraph/
├── libleidenalg/
├── leidenalg/
└── cluster_editing/
```

Native libraries required by the customized Leiden implementation are installed to:

```text
~/local/
```

### Activate the environment

After the setup has completed successfully, activate the virtual environment with:

```bash
source .venv/bin/activate
```

The activation script also configures the library paths required by the customized Leiden implementation.

## Data Generation

The experiments in this repository use two different groups of synthetic datasets:

- synthetic Powerlaw Cluster and Erdős–Rényi graphs used for the development and configuration of the MDGP local-search heuristic,
- Gaussian random partition graphs with a known ground-truth partition used to compare the resulting graph partitions with an underlying community structure.

All datasets can be generated directly from the scripts contained in the repository.

```bash
python scripts/generate_data.py
python scripts/generate_ground_truth_data.py
```

## Experiment 1: Start Partitions

The first experiment evaluates the different start-partition algorithms.

By default, the script only runs on the small graph instances:

```bash
python scripts/experiment1_baseline.py
```

To run the experiment on both small and large instances, explicitly set the data root to the complete generated dataset:

```bash
python scripts/experiment1_baseline.py \
    --data-root data/generated
```

The results are written to:

```text
results/experiment1/
```