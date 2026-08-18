#!/usr/bin/env bash

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORIES_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_DIR="$HOME/local"

echo "Repository directory: $REPOSITORIES_DIR"
echo "Local installation directory: $LOCAL_DIR"

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------

for command in git python3 uv cmake; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Error: '$command' is required but not installed."
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------

cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing Python dependencies..."
uv pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# Native igraph 1.0.0
# ---------------------------------------------------------------------------

cd "$REPOSITORIES_DIR"

if [ ! -d "igraph" ]; then
    echo "Cloning igraph..."
    git clone https://github.com/igraph/igraph.git
fi

cd igraph
git checkout 1.0.0

rm -rf build
mkdir build
cd build

echo "Building igraph..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$LOCAL_DIR" \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build . -j 8
cmake --install .

# ---------------------------------------------------------------------------
# Custom libleidenalg
# ---------------------------------------------------------------------------

cd "$REPOSITORIES_DIR"

if [ ! -d "libleidenalg" ]; then
    echo "Cloning custom libleidenalg..."
    git clone https://github.com/lulufka/libleidenalg.git
fi

cd libleidenalg

rm -rf build
mkdir build
cd build

echo "Building libleidenalg..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$LOCAL_DIR"

cmake --build . -j 8
cmake --install .

# ---------------------------------------------------------------------------
# Environment variables required by custom leidenalg
# ---------------------------------------------------------------------------

export CPLUS_INCLUDE_PATH="$LOCAL_DIR/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$LOCAL_DIR/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="$LOCAL_DIR/lib:${LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$LOCAL_DIR/lib:${DYLD_LIBRARY_PATH:-}"

# Store variables in virtual environment activation script so that they are
# also available after setup.sh has finished.
ACTIVATE_FILE="$SCRIPT_DIR/.venv/bin/activate"

if ! grep -q "MDGP_LOCAL_LIBRARIES" "$ACTIVATE_FILE"; then
    cat >> "$ACTIVATE_FILE" <<EOF

# MDGP_LOCAL_LIBRARIES
export CPLUS_INCLUDE_PATH="$LOCAL_DIR/include:\${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$LOCAL_DIR/include:\${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="$LOCAL_DIR/lib:\${LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$LOCAL_DIR/lib:\${DYLD_LIBRARY_PATH:-}"
EOF
fi

# ---------------------------------------------------------------------------
# Custom Python leidenalg
# ---------------------------------------------------------------------------

cd "$REPOSITORIES_DIR"

if [ ! -d "leidenalg" ]; then
    echo "Cloning custom leidenalg..."
    git clone https://github.com/lulufka/leidenalg.git
fi

cd leidenalg

echo "Installing custom leidenalg..."
uv pip install -v .

# ---------------------------------------------------------------------------
# KaPoCE
# ---------------------------------------------------------------------------

cd "$REPOSITORIES_DIR"

if [ ! -d "cluster_editing" ]; then
    echo "Cloning KaPoCE..."
    git clone --recursive https://github.com/lulufka/cluster_editing.git
fi

cd cluster_editing

rm -rf build
mkdir build
cd build

echo "Building KaPoCE..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build . -j 8

# ---------------------------------------------------------------------------
# Local project configuration
# ---------------------------------------------------------------------------

cd "$SCRIPT_DIR"

cat > config.local.json <<EOF
{
  "kapoce_executable": "$REPOSITORIES_DIR/cluster_editing/build/heuristic",
  "kapoce_config": "$REPOSITORIES_DIR/cluster_editing/config/fast.ini"
}
EOF

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

echo
echo "Checking installation..."

python - <<'PY'
import leidenalg

assert hasattr(leidenalg, "MDGPVertexPartition"), (
    "Custom MDGPVertexPartition is not available."
)

print("Custom leidenalg: OK")
PY

python - <<'PY'
import networkx as nx

from dense_graph_partition.adapters.kapoce import kapoce_partition
from dense_graph_partition.config import load_kapoce_config

config = load_kapoce_config()

assert config.executable_path.exists(), (
    f"KaPoCE executable not found: {config.executable_path}"
)
assert config.config_path.exists(), (
    f"KaPoCE config not found: {config.config_path}"
)

graph = nx.karate_club_graph()

partition = kapoce_partition(
    graph,
    executable_path=config.executable_path,
    config_path=config.config_path,
)

assert sum(len(cluster) for cluster in partition) == graph.number_of_nodes()

print("KaPoCE: OK")
PY

echo
echo "Installation completed successfully."
echo
echo "Activate the environment with:"
echo "  source \"$SCRIPT_DIR/.venv/bin/activate\""