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
# Get the directory of a sourced script relative to call depth
# depth=0 → the current script
# depth=1 → the script that sourced this one
# depth=2 → the script that sourced the script that sourced this one
# etc.
# ------------------------------
get_script_dir() {
    local depth="${1:-0}"  # default 0 if not provided
    local src_index=$((depth + 0))

    # Show all BASH_SOURCE entries for debugging
    echo -e "\n[DEBUG] BASH_SOURCE array:"
    for i in "${!BASH_SOURCE[@]}"; do
        echo "  [$i] = ${BASH_SOURCE[$i]}"
    done

    # Ensure we don't go out of bounds
    if [ "$src_index" -ge "${#BASH_SOURCE[@]}" ]; then
        src_index=$((${#BASH_SOURCE[@]} - 1))
        echo "[DEBUG] Requested depth $depth exceeds BASH_SOURCE length, using last index $src_index"
    fi

    local src="${BASH_SOURCE[$src_index]}"
    local dir
    dir="$(cd "$(dirname "$src")" && pwd)"

    echo "[DEBUG] Requested depth: $depth"
    echo "[DEBUG] Using BASH_SOURCE index: $src_index -> $src"
    echo "[DEBUG] Resolved directory: $dir"

    echo "$dir"
}

update_script_dir() {
    local depth="${1:-0}"
    SCRIPT_DIR="$(get_script_dir "$depth")"
    echo "[INFO] SCRIPT_DIR updated to: $SCRIPT_DIR (depth=$depth)"
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