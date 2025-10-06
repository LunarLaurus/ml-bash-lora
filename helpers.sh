#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

CUDA_SUPPORTED=("11.8" "12.1" "12.2" "12.3" "12.4" "12.6" "12.8" "12.9")

# ------------------------------
# Colors
# ------------------------------
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'

# ------------------------------
# Environment file
# ------------------------------
ML_ENV_FILE="$HOME/.ml_current_env"

# ------------------------------
# Script directory helpers
# ------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Return directory of a script
get_script_dir() {
    local dir
    dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "$dir"
}

check_env() {
    if [ ! -f "$ML_ENV_FILE" ]; then
        echo -e "${RED}No active ML environment set.${NC}"
        return 1
    fi
    CURRENT_ENV=$(cat "$ML_ENV_FILE")
    echo -e "${GREEN}Current active ML environment: $CURRENT_ENV${NC}"
    return 0
}

save_env() {
    echo "$1" > "$ML_ENV_FILE"
}

update_apt_cache() {
    echo "Updating apt package lists..."
    sudo apt-get update -qq
}

show_python_version() {
    if ! command -v python &>/dev/null; then
        echo -e "${RED}Python not found.${NC}"
    else 
        echo -e "${GREEN}Python version:${NC} $(python --version)"
    fi
}


detect_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo -e "${RED}NVIDIA driver not detected. Install first.${NC}"
        return 1
    fi
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_CC=$(nvidia-smi --query-gpu=compute_capability --format=csv,noheader | head -n1)
    echo -e "${GREEN}Detected GPU: $GPU_NAME (Compute Capability $GPU_CC)${NC}"
}