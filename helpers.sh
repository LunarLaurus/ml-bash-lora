#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

CUDA_SUPPORTED=("11.8" "12.1" "12.2" "12.3" "12.4" "12.6" "12.8" "12.9")

# ------------------------------
# Colors
# ------------------------------
# Standard Colors
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
# Bright / Bold variants
BRED='\033[1;31m'
BYELLOW='\033[1;33m'
BGREEN='\033[1;32m'
BBLUE='\033[1;34m'
BMAGENTA='\033[1;35m'
BCYAN='\033[1;36m'
BWHITE='\033[1;37m'
NC='\033[0m'  # No Color


# ------------------------------
# Environment file
# ------------------------------
ML_ENV_FILE="$HOME/.ml_current_env"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

# Generic logger: optional color as first argument
log() {
    local color="$1"
    shift
    if [ -n "$color" ]; then
        echo -e "${color}$*${NC}"
    else
        echo -e "$*"
    fi
}

# Wrappers
info()  { log "" "[INFO] $*"; }       # No default color
warn()  { log "$YELLOW" "[WARN] $*"; }
error() { log "$RED" "[ERROR] $*"; }

# ------------------------------
# Get the directory of a sourced script relative to call depth
# depth=0 → the current script
# depth=1 → the script that sourced this one
# depth=2 → the script that sourced the script that sourced this one
# etc.
# ------------------------------
get_script_dir() {
    local depth="${1:-0}"
    local max_index=$((${#BASH_SOURCE[@]} - 1))
    local idx=$(( depth <= max_index ? depth : max_index ))
    local src="${BASH_SOURCE[$idx]}"
    local dir
    dir="$(cd "$(dirname "$src")" && pwd)"
    echo "$dir"
}

# ------------------------------
# Update global SCRIPT_DIR
# ------------------------------
update_script_dir() {
    local depth="${1:-0}"
    
    SCRIPT_DIR="$(get_script_dir "$depth")"
    
    # Logging
    info -e "SCRIPT_DIR updated to: $SCRIPT_DIR (depth=$depth)"
}

check_env() {
    if [ ! -f "$ML_ENV_FILE" ]; then
        error -e "No active ML environment set."
        return 1
    fi
    CURRENT_ENV=$(cat "$ML_ENV_FILE")
    info -e "${GREEN}Current active ML environment: $CURRENT_ENV${NC}"
    return 0
}

error_no_env() {
    error -e "No active ML conda environment found in $ML_ENV_FILE."
    warn "Run the Conda / ML Environment menu and create/activate an environment first."
}

update_apt_cache() {
    info "Updating apt package lists..."
    sudo apt-get update -qq
}

detect_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        error -e "NVIDIA driver not detected. Install first."
        return 1
    fi
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_CC=$(nvidia-smi --query-gpu=compute_capability --format=csv,noheader 2>/dev/null | head -n1)
    if [ -z "$GPU_CC" ]; then
        warn "Unable to query 'compute_capability', trying 'compute_cap'."
        GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
    fi
    If still empty, warn the user
    if [ -z "$GPU_CC" ]; then
        warn "Could not detect GPU compute capability."
    else
        info "GPU compute capability: $GPU_CC"
    fi
    info -e "${GREEN}Detected GPU: $GPU_NAME (Compute Capability $GPU_CC)${NC}"
}


# Returns the PyTorch CUDA index string (e.g., cu118)
# Returns cu0 if no CUDA detected or no mapping exists
get_cu_index() {
    local idx
    idx=$(get_cuda_version_index)   # call the function that returns numeric index
    printf 'cu%s' "$idx"
}

# Returns the PyTorch CUDA index string (e.g., 118) based on CUDA_VER
# Returns 0 string if no CUDA detected or no mapping exists
get_cuda_version_index() {
    local cuda_ver="${CUDA_VER:-}"
    local cuidx="0"  # default to 0 if nothing found
    
    # Extract major.minor
    if [ -n "$cuda_ver" ]; then
        cuda_ver=$(printf '%s' "$cuda_ver" | grep -oE '^[0-9]+\.[0-9]+')
    fi
    
    # Mapping from CUDA version to cu index
    declare -A CUDA_TO_CUIDX=(
        ["11.8"]="118"
        ["12.1"]="121"
        ["12.2"]="122"
        ["12.3"]="123"
        ["12.4"]="124"
        ["12.6"]="126"
        ["12.8"]="128"
        ["12.9"]="129"
    )
    
    # Look up index; if missing, keep default 0
    cuidx="${CUDA_TO_CUIDX[$cuda_ver]:-$cuidx}"
    
    # Return value
    printf '%s' "$cuidx"
}

update_torch_index_url() {
    # Get PyTorch CUDA index (e.g., cu118, cu128)
    local cuidx
    cuidx=$(get_cu_index)
    
    if [[ -z "$cuidx" || "$cuidx" == "cu0" ]]; then
        warn -e "${YELLOW}No mapping for detected CUDA ${CUDA_VER:-unknown}. Defaulting to CPU wheel.${NC}" 2>/dev/null || true
        TORCH_INDEX_URL=""
    else
        TORCH_INDEX_URL="https://download.pytorch.org/whl/${cuidx}"
    fi
    
    export TORCH_INDEX_URL
    return 0
}

