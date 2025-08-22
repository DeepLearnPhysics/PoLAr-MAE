#!/bin/bash

# PoLAr-MAE Installation Script
# This script:
# 1. Creates a conda environment from environment.yml
# 2. Activates the environment
# 3. Installs pytorch3d from source with proper CUDA compilation
# 4. Installs cnms extension
# 5. Installs the main polarmae package

set -e  # exit on error

# default
MAX_JOBS=${MAX_JOBS:-4}
echo "Using MAX_JOBS=$MAX_JOBS for compilation"

command_exists() {
    command -v "$1" &> /dev/null
}

# try mamba first, then conda
if command_exists mamba; then
    CONDA_CMD="mamba"
elif command_exists conda; then
    CONDA_CMD="conda"
else
    echo "Error: Neither mamba nor conda command found. Please install conda or mamba first."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# create conda environment w/ environment.yml
echo "[1/5] Creating conda environment from environment.yml..."

# Check if the environment already exists
if $CONDA_CMD env list | grep -q "^polarmae "; then
    echo "Environment 'polarmae' already exists. Skipping..."
else
    $CONDA_CMD env create -f environment.yml --verbose
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create conda environment."
        exit 1
    fi
    echo "Conda environment 'polarmae' created successfully."
fi

echo "[2/5] Activating conda environment..."
# Source the conda initialization script to make conda/mamba commands available in this script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
# Also support mamba if that's what we're using
if [ "$CONDA_CMD" = "mamba" ]; then
    # Some systems have mamba in the libmamba-activate.sh file
    if [ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/mamba.sh"
    elif [ -f "$CONDA_BASE/etc/profile.d/libmamba-activate.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/libmamba-activate.sh"
    fi
fi

# Now activate the environment
conda activate polarmae
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'polarmae'."
    exit 1
fi
echo "Conda environment activated."

# build pytorch3d from source
echo "[3/5] Installing pytorch3d from source..."
cd "$SCRIPT_DIR/extensions/pytorch3d"
if [ ! -d "$SCRIPT_DIR/extensions/pytorch3d" ]; then
    echo "Error: pytorch3d directory not found at $SCRIPT_DIR/extensions/pytorch3d."
    echo "Clone the PoLAr-MAE repository and submodules using:"
    echo "    git clone --recurse-submodules https://github.com/DeepLearnPhysics/PoLAr-MAE.git"
    echo "and try again."
    exit 1
fi

export MAX_JOBS
python setup.py install
if [ $? -ne 0 ]; then
    echo "Error: Failed to install pytorch3d."
    exit 1
fi
echo "pytorch3d installed successfully."

# install cnms extension
echo "[4/5] Installing cnms extension..."
cd "$SCRIPT_DIR/extensions/cnms"

export MAX_JOBS
python setup.py install
if [ $? -ne 0 ]; then
    echo "Error: Failed to install cnms."
    exit 1
fi
echo "cnms installed successfully."

echo "[5/5] Installing polarmae..."
cd "$SCRIPT_DIR"
pip install -e .
if [ $? -ne 0 ]; then
    echo "Error: Failed to install polarmae."
    exit 1
fi
echo ""
echo "Installation complete. Activate the environment with:"
echo ""
echo "$CONDA_CMD activate polarmae"
echo ""
