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

## Running the Experiments

The experiments correspond to the following chapters and sections of the thesis:

| Experiment                            | Thesis chapter / section |
|---------------------------------------| --- |
| Experiment 1: Initial Partitions      | Section 5.4 — Initial Partitions |
| Experiment 2: Local Search            | Chapter 6 — Local-Search Heuristic |
| Experiment 3: Comparison              | Chapter 7 — Comparison with Existing Methods |
| Experiment 4: Ground-Truth Comparison | Chapter 8 — Experiments on Ground-Truth Data |

Experiment 2 is further divided into the following sub-experiments:

| Sub-experiment | Thesis section |
| --- | --- |
| `move_strategies` | Section 6.2 — Comparison of Move Strategies |
| `repeated_plateau_search` | Section 6.3 — Effect of Repeated Plateau Searches |
| `zero_gain_limit` | Section 6.4 — Effect of the Zero-Gain Limit |
| `initial_partition` | Section 6.5 — Effect of the Initial Partition |
| `additional_local_search_operator` | Section 6.6 — Combination with Additional Local-Search Operators |
| `independent_runs` | Section 6.7 — Effect of the Number of Independent Runs |

All experiment scripts support the following common arguments.

By default, experiments are run only on the small graph instances. To include both small and large instances, use:

```bash
--data-root data/generated
```

By default, the experiments run sequentially with one worker. Parallel execution can be enabled with:

```bash
--workers 8
```

## Experiment 1: Initial Partitions

The first experiment evaluates the different start-partition algorithms.

```bash
python scripts/experiment1_baseline.py
```

The results are written to:

```text
results/experiment1/
```

The corresponding analysis notebook is:

```text
notebooks/experiment1/experiment1_baseline_anaylsis.ipynb
```

## Experiment 2: Local Search

The second experiment evaluates the local-search configurations used to develop the final MDGP heuristic.

The individual sub-experiments can be executed via the `--experiment` argument.

### Move strategies

```bash
python scripts/experiment2_local_search.py \
    --experiment move_strategies
```

### Repeated plateau search

```bash
python scripts/experiment2_local_search.py \
    --experiment repeated_plateau_search
```

### Zero-gain limit

```bash
python scripts/experiment2_local_search.py \
    --experiment zero_gain_limit
```

### Initial partitions

```bash
python scripts/experiment2_local_search.py \
    --experiment initial_partition
```

### Additional local-search operators

```bash
python scripts/experiment2_local_search.py \
    --experiment additional_local_search_operator
```

### Independent runs

```bash
python scripts/experiment2_local_search.py \
    --experiment independent_runs
```

The results are written to:

```text
results/experiment2/<experiment>/
```

The corresponding analysis notebooks are located at:

```text
notebooks/experiment2/<experiment>_analysis.ipynb
```

## Experiment 3: Comparison

The third experiment compares the final MDGP heuristic with the considered comparison methods.

```bash
python scripts/experiment3_comparison.py
```

The results are written to:

```text
results/experiment3/
```

The corresponding analysis notebook is:

```text
notebooks/experiment3/experiment3_comparison.ipynb
```

## Experiment 4: Ground-Truth Comparison

The fourth experiment evaluates the considered algorithms on synthetic graphs with a known ground-truth partition.

```bash
python scripts/experiment4_ground_truth.py
```

The results are written to:

```text
results/experiment4/
```

The corresponding analysis notebook is:

```text
notebooks/experiment4/experiment4_ground_truth.ipynb
```

### Partition Visualization

The partitions obtained in Experiment 4 can be visualized together with the corresponding ground-truth partition.

```bash
python scripts/compare_partitions.py <path-to-ground-truth-instance>
```

The plots are written to:

```text
results/plots/
```