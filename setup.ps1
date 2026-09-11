$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoriesDir = Split-Path -Parent $ScriptDir
$LocalDir = Join-Path $env:USERPROFILE "local"
$VcpkgDir = Join-Path $RepositoriesDir "vcpkg"

Write-Host "Repository directory: $RepositoriesDir"
Write-Host "Local installation directory: $LocalDir"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

function Require-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found."
    }
}

function Remove-BuildDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (Test-Path $Path) {
        Remove-Item -Recurse -Force $Path
    }

    New-Item -ItemType Directory -Path $Path | Out-Null
}

function Clone-Repository {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,

        [Parameter(Mandatory = $true)]
        [string]$Directory,

        [switch]$Recursive
    )

    if (-not (Test-Path $Directory)) {
        Write-Host "Cloning $Url..."

        if ($Recursive) {
            git clone --recursive $Url $Directory
        }
        else {
            git clone $Url $Directory
        }
    }
}

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------

Write-Host
Write-Host "Checking prerequisites..."

Require-Command git
Require-Command cmake
Require-Command python

# CMake will normally discover Visual Studio automatically.
# This check gives an earlier error when no suitable compiler is installed.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

if (-not (Test-Path $vswhere)) {
    throw @"
Visual Studio Build Tools were not found.

Install Visual Studio Build Tools 2022 with:
  Desktop development with C++

Then run this script again.
"@
}

$vsInstallationPath = & $vswhere `
    -latest `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath

if (-not $vsInstallationPath) {
    throw @"
The Visual C++ build tools were not found.

Open Visual Studio Installer and install:
  Desktop development with C++

Then run this script again.
"@
}

Write-Host "Visual Studio installation: $vsInstallationPath"

# ---------------------------------------------------------------------------
# uv
# ---------------------------------------------------------------------------

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host
    Write-Host "uv was not found. Installing uv..."

    powershell -ExecutionPolicy ByPass -Command `
        "irm https://astral.sh/uv/install.ps1 | iex"

    $uvDir = Join-Path $env:USERPROFILE ".local\bin"
    $env:Path = "$uvDir;$env:Path"

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installation completed, but uv could not be found."
    }
}

Write-Host "uv: OK"

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------

Set-Location $ScriptDir

if (-not (Test-Path ".venv")) {
    Write-Host
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$VenvActivate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"

Write-Host
Write-Host "Installing Python dependencies..."
uv pip install --python $VenvPython -e ".[dev]"

# ---------------------------------------------------------------------------
# vcpkg and Boost
# ---------------------------------------------------------------------------

Set-Location $RepositoriesDir

Clone-Repository `
    -Url "https://github.com/microsoft/vcpkg.git" `
    -Directory $VcpkgDir

$VcpkgExe = Join-Path $VcpkgDir "vcpkg.exe"

if (-not (Test-Path $VcpkgExe)) {
    Write-Host
    Write-Host "Bootstrapping vcpkg..."
    & (Join-Path $VcpkgDir "bootstrap-vcpkg.bat")
}

Write-Host
Write-Host "Installing Boost program_options..."
& $VcpkgExe install boost-program-options:x64-windows

$VcpkgToolchain = Join-Path $VcpkgDir "scripts\buildsystems\vcpkg.cmake"
$VcpkgBin = Join-Path $VcpkgDir "installed\x64-windows\bin"

# ---------------------------------------------------------------------------
# Native igraph 1.0.0
# ---------------------------------------------------------------------------

$IgraphDir = Join-Path $RepositoriesDir "igraph"

Clone-Repository `
    -Url "https://github.com/igraph/igraph.git" `
    -Directory $IgraphDir

Set-Location $IgraphDir

git fetch --tags
git checkout 1.0.0

$IgraphBuild = Join-Path $IgraphDir "build"
Remove-BuildDirectory $IgraphBuild

Write-Host
Write-Host "Building igraph..."

cmake `
    -S $IgraphDir `
    -B $IgraphBuild `
    -A x64 `
    -DCMAKE_INSTALL_PREFIX="$LocalDir" `
    -DBUILD_SHARED_LIBS=ON `
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build $IgraphBuild --config Release --parallel
cmake --install $IgraphBuild --config Release

# ---------------------------------------------------------------------------
# Custom libleidenalg
# ---------------------------------------------------------------------------

$LibLeidenalgDir = Join-Path $RepositoriesDir "libleidenalg"

Clone-Repository `
    -Url "https://github.com/lulufka/libleidenalg.git" `
    -Directory $LibLeidenalgDir

$LibLeidenalgBuild = Join-Path $LibLeidenalgDir "build"
Remove-BuildDirectory $LibLeidenalgBuild

Write-Host
Write-Host "Building custom libleidenalg..."

cmake `
    -S $LibLeidenalgDir `
    -B $LibLeidenalgBuild `
    -A x64 `
    -DCMAKE_INSTALL_PREFIX="$LocalDir" `
    -DCMAKE_PREFIX_PATH="$LocalDir"

cmake --build $LibLeidenalgBuild --config Release --parallel
cmake --install $LibLeidenalgBuild --config Release

# ---------------------------------------------------------------------------
# Environment variables for native libraries
# ---------------------------------------------------------------------------

$LocalInclude = Join-Path $LocalDir "include"
$LocalLib = Join-Path $LocalDir "lib"
$LocalBin = Join-Path $LocalDir "bin"

$env:INCLUDE = "$LocalInclude;$env:INCLUDE"
$env:LIB = "$LocalLib;$env:LIB"
$env:PATH = "$LocalBin;$VcpkgBin;$env:PATH"
$env:CMAKE_PREFIX_PATH = $LocalDir

# Store library paths in the virtual environment activation script.
$ActivationMarker = "# MDGP_LOCAL_LIBRARIES"

if (-not (Select-String -Path $VenvActivate -Pattern $ActivationMarker -Quiet)) {
    Add-Content -Path $VenvActivate -Value @"

$ActivationMarker
`$env:INCLUDE = "$LocalInclude;`$env:INCLUDE"
`$env:LIB = "$LocalLib;`$env:LIB"
`$env:PATH = "$LocalBin;$VcpkgBin;`$env:PATH"
`$env:CMAKE_PREFIX_PATH = "$LocalDir"
"@
}

# ---------------------------------------------------------------------------
# Custom Python leidenalg
# ---------------------------------------------------------------------------

$LeidenalgDir = Join-Path $RepositoriesDir "leidenalg"

Clone-Repository `
    -Url "https://github.com/lulufka/leidenalg.git" `
    -Directory $LeidenalgDir

Set-Location $LeidenalgDir

Write-Host
Write-Host "Installing custom leidenalg..."

uv pip install `
    --python $VenvPython `
    -v .

# ---------------------------------------------------------------------------
# KaPoCE
# ---------------------------------------------------------------------------

$ClusterEditingDir = Join-Path $RepositoriesDir "cluster_editing"

Clone-Repository `
    -Url "https://github.com/lulufka/cluster_editing.git" `
    -Directory $ClusterEditingDir `
    -Recursive

$ClusterEditingBuild = Join-Path $ClusterEditingDir "build"
Remove-BuildDirectory $ClusterEditingBuild

Write-Host
Write-Host "Building KaPoCE..."

cmake `
    -S $ClusterEditingDir `
    -B $ClusterEditingBuild `
    -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 `
    -DCMAKE_TOOLCHAIN_FILE="$VcpkgToolchain"

cmake --build $ClusterEditingBuild --config Release --parallel

# Locate the generated executable.
$KapoceExecutable = Get-ChildItem `
    -Path $ClusterEditingBuild `
    -Filter "heuristic.exe" `
    -Recurse |
    Select-Object -First 1

if (-not $KapoceExecutable) {
    throw "KaPoCE executable heuristic.exe was not found."
}

$KapoceConfig = Join-Path $ClusterEditingDir "config\fast.ini"

if (-not (Test-Path $KapoceConfig)) {
    throw "KaPoCE configuration was not found: $KapoceConfig"
}

# ---------------------------------------------------------------------------
# Local project configuration
# ---------------------------------------------------------------------------

Set-Location $ScriptDir

$KapoceExecutablePath = $KapoceExecutable.FullName.Replace("\", "/")
$KapoceConfigPath = $KapoceConfig.Replace("\", "/")

$config = @"
{
  "kapoce_executable": "$KapoceExecutablePath",
  "kapoce_config": "$KapoceConfigPath"
}
"@

Set-Content `
    -Path (Join-Path $ScriptDir "config.local.json") `
    -Value $config `
    -Encoding UTF8

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

Write-Host
Write-Host "Checking custom Leiden installation..."

& $VenvPython -c @"
import leidenalg

assert hasattr(leidenalg, "MDGPVertexPartition"), (
    "Custom MDGPVertexPartition is not available."
)

print("Custom leidenalg: OK")
"@

Write-Host
Write-Host "Checking KaPoCE installation..."

& $VenvPython -c @"
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
"@

Write-Host
Write-Host "Installation completed successfully."
Write-Host
Write-Host "Activate the environment with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"