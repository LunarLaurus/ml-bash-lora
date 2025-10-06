#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# ==============================
# ML/LoRA + RAG Setup Script
# Auto-detect GPU/CUDA/PyTorch
# Ubuntu 24.04 / RTX 4000+
# ==============================

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'
ML_ENV_FILE="$HOME/.ml_current_env"
NVIDIA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
CUDA_SUPPORTED=("11.8" "12.1" "12.2" "12.3" "12.4" "12.6" "12.8" "12.9")

# ------------------------------
# Helper Functions
# ------------------------------

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

# ------------------------------
# Detect GPU and CUDA
# ------------------------------

detect_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo -e "${RED}NVIDIA driver not detected. Install first.${NC}"
        return 1
    fi
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_CC=$(nvidia-smi --query-gpu=compute_capability --format=csv,noheader | head -n1)
    echo -e "${GREEN}Detected GPU: $GPU_NAME (Compute Capability $GPU_CC)${NC}"
}

# Convenience wrapper:
# - lists installations, prompts selection if >1 (default highest),
# - updates /usr/local/cuda symlink,
# - persists env if asked (flag --persist)
# usage: detect_cuda [--persist]
detect_cuda() {
    detect_cuda_select || return 1

    # show nvcc info if available
    if command -v nvcc &>/dev/null; then
        echo
        show_nvcc_version
    else
        echo -e "${RED}nvcc not found in PATH after selection.${NC}"
    fi
}

# ------------------------------
# List available CUDA versions for current Ubuntu release
# ------------------------------
list_available_cuda_versions() {
    # List all CUDA packages from apt-cache
    local detected_versions=()
    while IFS= read -r line; do
        if [[ $line =~ cuda-([0-9]+)-([0-9]+) ]]; then
            detected_versions+=("${BASH_REMATCH[1]}.${BASH_REMATCH[2]}")
        fi
    done < <(apt-cache search '^cuda-[0-9]+-[0-9]+$')

    # Remove duplicates and sort
    IFS=$'\n' detected_versions=($(sort -Vu <<<"${detected_versions[*]}"))
    unset IFS

    # Show detected versions
    echo -e "\n${GREEN}Detected CUDA versions from apt:${NC}"
    if [ "${#detected_versions[@]}" -eq 0 ]; then
        echo "  (none detected via apt-cache)"
    else
        for v in "${detected_versions[@]}"; do
            echo "  - $v"
        done
    fi

    # Show default supported versions
    echo -e "\n${GREEN}Default supported CUDA versions:${NC}"
    for s in "${CUDA_SUPPORTED[@]}"; do
        echo "  - $s"
    done

    # Ask if user wants to override default list
    local override_supported=0
    read -rp "Do you want to override the default supported CUDA list with detected versions? [y/N]: " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        override_supported=1
    fi

    # Filter versions if not overriding
    local versions=("${detected_versions[@]}")
    if [[ $override_supported -eq 0 ]]; then
        local filtered=()
        for v in "${versions[@]}"; do
            for s in "${CUDA_SUPPORTED[@]}"; do
                [[ "$v" == "$s" ]] && filtered+=("$v")
            done
        done
        versions=("${filtered[@]}")
    fi

    # Show final available list
    echo -e "\n${GREEN}Final available CUDA versions:${NC}"
    if [ "${#versions[@]}" -eq 0 ]; then
        echo "  (none available)"
    else
        for v in "${versions[@]}"; do
            echo "  - $v"
        done
    fi
}

# Return list of candidate CUDA directories (global array: CUDA_CANDIDATES)
detect_cuda_list() {
    CUDA_CANDIDATES=()
    # Globs to check; add other locations here if you have nonstandard installs
    for g in /usr/local/cuda-* /opt/cuda-* /usr/local/cuda* /opt/cuda*; do
        for p in $g; do
            [ -d "$p" ] || continue
            CUDA_CANDIDATES+=("$(readlink -f "$p")")
        done
    done

    # If /usr/local/cuda exists, include its real target (avoid duplicate)
    if [ -e /usr/local/cuda ]; then
        t=$(readlink -f /usr/local/cuda 2>/dev/null || true)
        [ -n "$t" ] && [ -d "$t" ] && CUDA_CANDIDATES+=("$t")
    fi

    # dedupe preserving order
    if [ "${#CUDA_CANDIDATES[@]}" -gt 0 ]; then
        local uniq=()
        for p in "${CUDA_CANDIDATES[@]}"; do
            [[ " ${uniq[*]} " == *" $p "* ]] || uniq+=("$p")
        done
        CUDA_CANDIDATES=("${uniq[@]}")
    fi
}

# Map each candidate path -> parsed version string (populates CUDA_MAP as "ver|path" lines)
_build_version_map() {
    CUDA_MAP=()
    for dir in "${CUDA_CANDIDATES[@]}"; do
        base="$(basename "$dir")"
        # attempt to parse version from directory name
        ver="$(printf '%s' "$base" | sed -E 's/^cuda[-_]?//I; s/^cudatoolkit[-_]?//I; s/[^0-9.].*//')"
        # fallback: use nvcc from that dir if available
        if [ -z "$ver" ] && [ -x "$dir/bin/nvcc" ]; then
            ver="$("$dir/bin/nvcc" --version 2>/dev/null | grep -oE 'release [0-9]+(\.[0-9]+)*' | sed 's/release //; s/,//')"
        fi
        # if still empty, use full path as a final fallback to ensure sortable value (will appear last)
        [ -z "$ver" ] && ver="$dir"
        CUDA_MAP+=("$ver|$dir")
    done
}

# Show found CUDA installations, returns count
list_cuda_installations() {
    detect_cuda_list
    if [ "${#CUDA_CANDIDATES[@]}" -eq 0 ]; then
        echo -e "${RED}No CUDA installations found.${NC}"
        return 0
    fi

    _build_version_map

    echo -e "${GREEN}Found CUDA installations:${NC}"
    # sort by version (semantic sort -V) and print
    IFS=$'\n' sorted=($(printf '%s\n' "${CUDA_MAP[@]}" | sort -t'|' -k1,1 -V))
    local i=1
    for entry in "${sorted[@]}"; do
        ver="${entry%%|*}"
        path="${entry#*|}"
        printf "  %2d) %s -> %s\n" "$i" "$ver" "$path"
        ((i++))
    done

    # expose sorted list externally if needed
    CUDA_MAP_SORTED=("${sorted[@]}")
    return "${#sorted[@]}"
}

# Interactively select CUDA (if >1), default to highest when Enter pressed
# After selection it sets CUDA_PATH, CUDA_VER (exported) and optionally makes /usr/local/cuda link and persists env
detect_cuda_select() {
    # Detect CUDA installations on disk
    detect_cuda_list
    _build_version_map

    if [ "${#CUDA_MAP[@]}" -eq 0 ]; then
        echo -e "${RED}No CUDA installations found on disk.${NC}"
        CUDA_VER=""
        CUDA_PATH=""
        return 1
    fi

    # Sort by version (semantic sort)
    IFS=$'\n' sorted=($(printf '%s\n' "${CUDA_MAP[@]}" | sort -t'|' -k1,1 -V))
    unset IFS
    CUDA_MAP_SORTED=("${sorted[@]}")

    # Determine default selection (highest version)
    default_index="${#CUDA_MAP_SORTED[@]}"
    highest="${CUDA_MAP_SORTED[-1]}"
    highest_ver="${highest%%|*}"
    highest_path="${highest#*|}"

    # Display available installations
    echo -e "${GREEN}Detected CUDA installations:${NC}"
    local i=1
    for entry in "${CUDA_MAP_SORTED[@]}"; do
        ver="${entry%%|*}"
        path="${entry#*|}"
        marker=""
        if [ "$i" -eq "$default_index" ]; then
            marker="(default)"
        fi
        printf "  %2d) %s -> %s %s\n" "$i" "$ver" "$path" "$marker"
        ((i++))
    done

    # Prompt user to select
    read -rp "Select CUDA version [default: highest]: " choice
    if [ -z "$choice" ]; then
        choice_index=$default_index
    elif ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#CUDA_MAP_SORTED[@]}" ]; then
        echo -e "${YELLOW}Invalid choice, using default.${NC}"
        choice_index=$default_index
    else
        choice_index=$choice
    fi

    # Set selection
    selected="${CUDA_MAP_SORTED[$((choice_index-1))]}"
    CUDA_VER="${selected%%|*}"
    CUDA_PATH="${selected#*|}"

    echo "Linking /usr/local/cuda -> $CUDA_PATH"
    # sudo ln -sfn "$CUDA_PATH" /usr/local/cuda

    export CUDA_PATH
    export CUDA_VER
	export PATH="$CUDA_PATH/bin:$PATH"
	export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"

    # Persist environment
    set_cuda_env_persistent "$CUDA_PATH"

    echo -e "${GREEN}CUDA ${CUDA_VER} selected and persisted.${NC}"
}

set_cuda_env_persistent() {
    local target="${1:-/usr/local/cuda}"
    local content="# CUDA environment - auto-generated
export CUDA_PATH=\"$target\"
export PATH=\"\$CUDA_PATH/bin:\$PATH\"
export LD_LIBRARY_PATH=\"\$CUDA_PATH/lib64:\${LD_LIBRARY_PATH:-}\"
"

    # Always write to the user's shell rc (~/.bashrc) to persist for normal user
    local rc="$HOME/.bashrc"

    # Avoid duplicate append
    if ! grep -q "## CUDA environment - auto-generated" "$rc" 2>/dev/null; then
        printf "\n## CUDA environment - auto-generated\n" >> "$rc"
        printf '%s\n' "$content" >> "$rc"
        echo -e "${GREEN}Appended CUDA env to $rc (applies at next login).${NC}"
    fi

    # Apply immediately to current shell (works for normal user)
    export CUDA_PATH="$target"
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
    echo -e "${GREEN}CUDA environment variables set in current shell.${NC}"
}

# Quick helper: ensure a valid CUDA is selected and persist it if missing from PATH.
# If PATH doesn't contain /usr/local/cuda/bin or nvcc not resolving there, offer to persist.
ensure_cuda_in_path_and_persist() {
    # run detection/selection if nvcc not in PATH
    if ! command -v nvcc &>/dev/null; then
        echo -e "${YELLOW}nvcc not found in PATH. Running detection...${NC}"
        detect_cuda
        return $?
    fi

    # check nvcc path
    nvcc_path="$(command -v nvcc)"
    nvcc_real="$(readlink -f "$nvcc_path" 2>/dev/null || true)"

    if [ -L /usr/local/cuda ]; then
        cur_target="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
        if [[ "$nvcc_real" == "$cur_target"* ]]; then
            echo -e "${GREEN}nvcc already resolves to /usr/local/cuda installation.${NC}"
            return 0
        fi
    fi

    echo -e "${YELLOW}nvcc found at: $nvcc_path${NC}"
    read -rp "Set /usr/local/cuda to this nvcc's parent and persist env? [y/N]: " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
        cuda_root="$(dirname "$(dirname "$nvcc_real")")"
        echo "Linking /usr/local/cuda -> $cuda_root"
        sudo ln -sfn "$cuda_root" /usr/local/cuda
        set_cuda_env_persistent "$cuda_root"
        echo -e "${GREEN}Done. Re-login for persistent env to take effect.${NC}"
    else
        echo "No changes made."
    fi
}


# ------------------------------
# Suggest and Install CUDA
# ------------------------------
install_cuda() {
    ensure_nvidia_repo || return 1

	list_available_cuda_versions
    echo -e "${GREEN}Suggested CUDA versions for optimal PyTorch compatibility:${NC}"
    echo " - 12.9 / 12.8 / 12.6 / 12.4 / 12.3 / 12.2 / 12.1: Recent GPUs & recent PyTorch builds"
    echo " - 11.8: Maximum compatibility for older PyTorch releases / older toolchains"
    echo "Supported: ${CUDA_SUPPORTED[*]}"
    read -rp "Enter desired CUDA version (major.minor) [12.2]: " CUDA_INPUT
    CUDA_INPUT=${CUDA_INPUT:-12.2}

    # Normalize input (12-2, 12.2.0 -> 12.2)
    CUDA_INPUT="$(printf '%s' "$CUDA_INPUT" | sed -E 's/[-_]//g; s/^([0-9]+)\.([0-9]+).*$/\1.\2/')"

    # Validate version
    if ! [[ " ${CUDA_SUPPORTED[*]} " =~ " ${CUDA_INPUT} " ]]; then
        echo -e "${YELLOW}Requested version '$CUDA_INPUT' not in supported list. Defaulting to 12.2.${NC}"
        CUDA_INPUT="12.2"
    fi

    pkg_ver="${CUDA_INPUT//./-}"
    cuda_pkg="cuda-${pkg_ver}"

    echo -e "${GREEN}Installing CUDA package: $cuda_pkg${NC}"
    sudo apt-get install -y "$cuda_pkg" || {
        echo -e "${YELLOW}Initial install failed; retrying with -f install...${NC}"
        sudo apt-get -f install -y
        sudo apt-get install -y "$cuda_pkg" || {
            echo -e "${RED}Failed to install $cuda_pkg.${NC}"
            return 1
        }
    }

    echo -e "${GREEN}CUDA $CUDA_INPUT installed successfully.${NC}"
	echo -e "${GREEN}Run detect_cuda to select and persist this installation.${NC}"

}
# ------------------------------
# Select and install PyTorch wheel (auto-installs & validates CUDA <-> torch match)
# ------------------------------
select_pytorch_wheel() {
    # Ensure CUDA is detected and persisted if missing
    detect_cuda --persist || true

    # Use the Python from the active conda env
    PYTHON_CMD=$(which python)
    PIP_CMD="$PYTHON_CMD -m pip"

    # Get CUDA version (major.minor)
    cuda_ver_num="${CUDA_VER:-}"
    if [ -n "$cuda_ver_num" ]; then
        cuda_ver_num=$(printf '%s' "$cuda_ver_num" | grep -oE '^[0-9]+\.[0-9]+')
    fi

    # If no CUDA, default to CPU wheel
    if [ -z "$cuda_ver_num" ]; then
        echo -e "${YELLOW}No CUDA detected. Installing CPU-only PyTorch wheel.${NC}"
        SUGGESTED="torch torchvision torchaudio"
        echo -e "${GREEN}Installing: $SUGGESTED${NC}"
        $PIP_CMD install --upgrade $SUGGESTED || return 1
        echo -e "${GREEN}CPU PyTorch installed.${NC}"
        return 0
    fi

    # CUDA version -> PyTorch cu index
    declare -A CUDA_TO_CUIDX=(
        ["11.8"]="cu118"
        ["12.1"]="cu121"
        ["12.2"]="cu122"
        ["12.3"]="cu123"
        ["12.4"]="cu124"
        ["12.6"]="cu126"
        ["12.8"]="cu128"
        ["12.9"]="cu129"
    )
    declare -A CUIDX_TO_TORCH_VER
    for k in "${!CUDA_TO_CUIDX[@]}"; do
        CUIDX_TO_TORCH_VER[${CUDA_TO_CUIDX[$k]}]=$k
    done

    cuidx="${CUDA_TO_CUIDX[$cuda_ver_num]}"
    if [ -z "$cuidx" ]; then
        echo -e "${YELLOW}No mapping for detected CUDA $cuda_ver_num. Defaulting to CPU wheel.${NC}"
        SUGGESTED="torch torchvision torchaudio"
        $PIP_CMD install --upgrade $SUGGESTED || return 1
        return 0
    fi

    SUGGESTED="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${cuidx}"
    echo -e "${GREEN}Installing PyTorch for CUDA $cuda_ver_num using wheel index $cuidx...${NC}"
    $PIP_CMD install --upgrade $SUGGESTED || return 1

    # Validate installed torch
    TORCH_REPORTED="$($PYTHON_CMD -c 'import torch; v=getattr(torch.version,"cuda",None); print(v or "None")')"
    normalized_reported="$(printf '%s' "$TORCH_REPORTED" | sed -E 's/[^0-9.]//g' | grep -oE '^[0-9]+\.[0-9]+')"

    expected_torch_ver="${CUIDX_TO_TORCH_VER[$cuidx]}"
    if [ "$normalized_reported" = "$expected_torch_ver" ]; then
        echo -e "${GREEN}Success: Installed PyTorch matches CUDA $cuda_ver_num.${NC}"
    else
        echo -e "${RED}Warning: Installed PyTorch reports CUDA $normalized_reported, expected $expected_torch_ver.${NC}"
    fi
}

# ------------------------------
# Install NVIDIA drivers
# ------------------------------
install_drivers() {
    echo -e "${GREEN}Updating system and installing NVIDIA drivers...${NC}"
    sudo apt update && sudo apt upgrade -y
    sudo ubuntu-drivers devices
    sudo ubuntu-drivers autoinstall
    echo -e "${GREEN}Drivers installed. Reboot recommended before using GPU.${NC}"
}

# ------------------------------
# Install Miniforge (Conda)
# ------------------------------
install_conda() {
    echo -e "${GREEN}Installing Miniforge3...${NC}"
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh
    bash ~/miniforge.sh -b -p $HOME/miniforge
    eval "$($HOME/miniforge/bin/conda shell.bash hook)"
    conda init
    echo -e "${GREEN}Conda installed. Restart shell to use 'conda activate'.${NC}"
}

# ------------------------------
# Create ML Environment
# ------------------------------
create_ml_env() {
    # Ensure conda is available
    if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniforge/etc/profile.d/conda.sh"
    else
        echo -e "${RED}Conda not found. Install Miniforge first.${NC}"
        return 1
    fi

    # Prompt for environment name and Python version
    read -rp "Enter environment name [lora]: " ENV_NAME
    ENV_NAME=${ENV_NAME:-lora}
    read -rp "Enter Python version (3.10 or 3.11 recommended) [3.10]: " PY_VER
    PY_VER=${PY_VER:-3.10}

    # Check if environment exists
    if conda env list | grep -qw "$ENV_NAME"; then
        echo -e "${GREEN}Environment $ENV_NAME already exists.${NC}"
        read -rp "Do you want to (R)einstall packages, (S)kip, or (E)xit? [S]: " choice
        choice=${choice:-S}
        case "$choice" in
            R|r) echo "Reinstalling packages..." ;;
            S|s) return ;;
            E|e) exit 0 ;;
        esac
    else
        conda create -y -n "$ENV_NAME" python="$PY_VER"
        echo -e "${GREEN}Environment $ENV_NAME created.${NC}"
    fi

    # Activate environment
    conda activate "$ENV_NAME"
    save_env "$ENV_NAME"
    PYTHON_CMD=$(which python)
    PIP_CMD="$PYTHON_CMD -m pip"
    echo -e "${GREEN}Activated environment $ENV_NAME.${NC}"

    # Detect GPU and CUDA
    detect_gpu
    detect_cuda

    # Install PyTorch if missing
    if ! $PYTHON_CMD -c "import torch" &>/dev/null; then
        select_pytorch_wheel
    else
        echo -e "${GREEN}PyTorch already installed.${NC}"
    fi

    # Install LoRA/Hugging Face stack
    for pkg in transformers peft datasets accelerate bitsandbytes; do
        if ! $PYTHON_CMD -c "import $pkg" &>/dev/null; then
            echo -e "${GREEN}Installing $pkg...${NC}"
            $PIP_CMD install --upgrade "$pkg"
        else
            echo -e "${GREEN}$pkg already installed.${NC}"
        fi
    done

    # Optional RAG stack
    read -rp "Install RAG stack (faiss, sentence-transformers, langchain)? [y/N]: " rag
    if [[ "$rag" =~ ^[Yy]$ ]]; then
        for pkg in faiss-cpu sentence-transformers langchain; do
            if ! $PYTHON_CMD -c "import $pkg" &>/dev/null; then
                echo -e "${GREEN}Installing $pkg...${NC}"
                $PIP_CMD install "$pkg"
            else
                echo -e "${GREEN}$pkg already installed.${NC}"
            fi
        done
    fi

    # Validate environment
    validate_env
}

# ------------------------------
# Switch active ML environment
# ------------------------------
switch_env() {
    conda env list
    read -p "Enter the environment name to activate: " ENV_NAME
    if conda env list | grep -qw "$ENV_NAME"; then
        save_env "$ENV_NAME"
        echo -e "${GREEN}Environment $ENV_NAME is now active.${NC}"
        source "$HOME/miniforge/bin/activate" "$ENV_NAME"
    else
        echo -e "${RED}Environment $ENV_NAME does not exist.${NC}"
    fi
}

# ------------------------------
# Validation Function
# ------------------------------
validate_env() {
    check_env || return 1
    source "$HOME/miniforge/bin/activate" "$CURRENT_ENV"

    echo -e "${GREEN}Validating ML environment...${NC}"

    detect_gpu
    detect_cuda

    $PYTHON_CM - <<EOF
import torch
try:
    if torch.cuda.is_available():
        print("PyTorch sees GPU:", torch.cuda.get_device_name(0))
    else:
        print("Warning: PyTorch cannot detect GPU")
except Exception as e:
    print("Error checking PyTorch GPU:", e)
EOF

    $PYTHON_CM - <<EOF
try:
    import bitsandbytes as bnb
    print("bitsandbytes installed successfully")
except Exception as e:
    print("bitsandbytes not installed or misconfigured:", e)
EOF

    echo -e "${GREEN}Validation complete.${NC}"
}

# ------------------------------
# Remove ML Environment / Project
# ------------------------------
remove_ml_env() {
    conda env list
    read -p "Enter the environment name to remove: " ENV_NAME
    if conda env list | grep -qw "$ENV_NAME"; then
        read -p "Are you sure you want to permanently delete '$ENV_NAME'? [y/N]: " CONFIRM
        if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
            conda deactivate &>/dev/null
            conda env remove -n "$ENV_NAME"
            echo -e "${GREEN}Environment '$ENV_NAME' removed successfully.${NC}"

            # Clear active environment if it was the removed one
            if [ -f "$ML_ENV_FILE" ] && grep -qw "$ENV_NAME" "$ML_ENV_FILE"; then
                rm -f "$ML_ENV_FILE"
                echo -e "${GREEN}Active environment cleared.${NC}"
            fi
        else
            echo "Aborted environment removal."
        fi
    else
        echo -e "${RED}Environment '$ENV_NAME' does not exist.${NC}"
    fi
}

# ------------------------------
# Show Disk Usage
# ------------------------------
show_disk_usage() {
    echo -e "${GREEN}Disk Usage:${NC}"
    df -h | awk 'NR==1 || /^\/dev\//'
}

# ------------------------------
# Show NVIDIA GPU Status
# ------------------------------
show_nvidia_smi() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo -e "${RED}NVIDIA driver not detected.${NC}"
        return 1
    fi
    echo -e "${GREEN}NVIDIA GPU Status:${NC}"
    nvidia-smi
}

# ------------------------------
# Show nvcc Version
# ------------------------------
show_nvcc_version() {
	auto_detect_nvcc
    if ! command -v nvcc &>/dev/null; then
        echo -e "${RED}CUDA compiler (nvcc) not found.${NC}"
        return 1
    fi
    echo -e "${GREEN}nvcc / CUDA version:${NC}"
    nvcc --version
}

auto_detect_nvcc() {
    if command -v nvcc &>/dev/null; then
        nvcc_dir="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
        echo "nvcc detected at $nvcc_dir"
        read -rp "Use this CUDA and persist env? [y/N]: " ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            CUDA_PATH="$nvcc_dir"
            CUDA_VER="$("$CUDA_PATH/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
            # sudo ln -sfn "$CUDA_PATH" /usr/local/cuda
            set_cuda_env_persistent "$CUDA_PATH"
            echo -e "${GREEN}CUDA ${CUDA_VER} persisted.${NC}"
        fi
    else
        echo -e "${YELLOW}nvcc not found in PATH.${NC}"
    fi
}

# ------------------------------
# Show Python Version in Active Env
# ------------------------------
show_python_version() {
    if ! command -v python &>/dev/null; then
        echo -e "${RED}Python not found in active environment.${NC}"
        return 1
    fi
    echo -e "${GREEN}Python version:${NC} $(python --version)"
}

# ------------------------------
# Show Disk Usage of Conda Environments
# ------------------------------
env_disk_usage() {
    echo -e "${GREEN}Disk usage of Conda environments:${NC}"
    du -sh $HOME/miniforge/envs/* 2>/dev/null
}

# ------------------------------
# Ensure NVIDIA CUDA repo exists
# ------------------------------
ensure_nvidia_repo() {
    echo -e "${GREEN}Checking NVIDIA CUDA repository...${NC}"

    # If repo not found in apt sources, add it
    if ! grep -Rq "developer.download.nvidia.com" /etc/apt/sources.list*; then
        echo -e "${YELLOW}NVIDIA CUDA repo not found. Adding it...${NC}"

        # Install prerequisite for add-apt-repository
        sudo apt-get update
        sudo apt-get install -y software-properties-common wget gnupg ca-certificates

        # Add pin file
        sudo wget -q "${NVIDIA_REPO_URL}cuda-ubuntu2204.pin" -O /etc/apt/preferences.d/cuda-repository-pin-600

        # Add keyring
        sudo wget -qO /usr/share/keyrings/cuda-archive-keyring.gpg \
            "${NVIDIA_REPO_URL}cuda-archive-keyring.gpg"

        # Add repo
        echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] \
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
        | sudo tee /etc/apt/sources.list.d/cuda.list >/dev/null

        sudo apt-get update
    fi

    # Test if repo is reachable
    if ! curl -s --head --fail "$NVIDIA_REPO_URL" >/dev/null; then
        echo -e "${YELLOW}⚠ NVIDIA repo is not reachable.${NC}"
        return 1
    fi

    echo -e "${GREEN}NVIDIA repo is valid and reachable.${NC}"
}

update_apt_cache() {
    echo "Updating apt package lists..."
    sudo apt-get update -qq
}

list_cuda_candidates() {
    detect_cuda_list
    _build_version_map

    if [ "${#CUDA_MAP[@]}" -eq 0 ]; then
        echo -e "${YELLOW}No CUDA installations found on disk.${NC}"
        return 1
    fi

    echo -e "${GREEN}Detected CUDA installations:${NC}"
    IFS=$'\n' sorted=($(printf '%s\n' "${CUDA_MAP[@]}" | sort -t'|' -k1,1 -V))
    CUDA_MAP_SORTED=("${sorted[@]}")
    local i=1
    for entry in "${sorted[@]}"; do
        ver="${entry%%|*}"
        path="${entry#*|}"
        printf "  %2d) %s -> %s\n" "$i" "$ver" "$path"
        ((i++))
    done
}

select_and_persist_cuda() {
    list_cuda_candidates || return 1

    local default_index="${#CUDA_MAP_SORTED[@]}"  # highest version
    read -rp "Select CUDA version [default: highest]: " choice

    if [ -z "$choice" ]; then
        choice_index=$default_index
    else
        if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#CUDA_MAP_SORTED[@]}" ]; then
            echo -e "${YELLOW}Invalid choice, using default.${NC}"
            choice_index=$default_index
        else
            choice_index=$choice
        fi
    fi

    selected="${CUDA_MAP_SORTED[$((choice_index-1))]}"
    CUDA_VER="${selected%%|*}"
    CUDA_PATH="${selected#*|}"

    echo "Linking /usr/local/cuda -> $CUDA_PATH"
    sudo ln -sfn "$CUDA_PATH" /usr/local/cuda

    export CUDA_PATH
    export CUDA_VER
    export PATH="/usr/local/cuda/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

    set_cuda_env_persistent "$CUDA_PATH"
	# Apply immediately to current shell
	if [ -f /etc/profile.d/cuda.sh ]; then
		source /etc/profile.d/cuda.sh
	fi

    echo -e "${GREEN}CUDA ${CUDA_VER} selected and persisted.${NC}"
}

remove_cuda_version() {
    # Detect installed CUDA versions
    detect_cuda_list
    _build_version_map

    if [ "${#CUDA_MAP[@]}" -eq 0 ]; then
        echo -e "${RED}No CUDA installations found on disk.${NC}"
        return 1
    fi

    # Sort by version
    IFS=$'\n' sorted=($(printf '%s\n' "${CUDA_MAP[@]}" | sort -t'|' -k1,1 -V))
    unset IFS
    CUDA_MAP_SORTED=("${sorted[@]}")

    # Show installations
    echo -e "${GREEN}Installed CUDA versions:${NC}"
    local i=1
    for entry in "${CUDA_MAP_SORTED[@]}"; do
        ver="${entry%%|*}"
        path="${entry#*|}"
        printf "  %2d) %s -> %s\n" "$i" "$ver" "$path"
        ((i++))
    done

    # Prompt user for selection
    read -rp "Enter number of CUDA version to remove (or multiple comma-separated): " choices
    IFS=',' read -ra choice_arr <<< "$choices"

    for c in "${choice_arr[@]}"; do
        if ! [[ "$c" =~ ^[0-9]+$ ]] || [ "$c" -lt 1 ] || [ "$c" -gt "${#CUDA_MAP_SORTED[@]}" ]; then
            echo -e "${YELLOW}Skipping invalid choice: $c${NC}"
            continue
        fi

        selected="${CUDA_MAP_SORTED[$((c-1))]}"
        ver="${selected%%|*}"
        path="${selected#*|}"

        echo -e "${YELLOW}Removing CUDA $ver at $path ...${NC}"
        sudo rm -rf "$path"

        # Check if /usr/local/cuda points here, remove link if so
        if [ -L /usr/local/cuda ]; then
            cur_link=$(readlink -f /usr/local/cuda)
            if [ "$cur_link" = "$path" ]; then
                echo "Removing /usr/local/cuda symlink pointing to removed version"
                sudo rm -f /usr/local/cuda
            fi
        fi
    done

    echo -e "${GREEN}CUDA removal complete.${NC}"
}
remove_obsolete_cuda_versions() {
    # Detect installed CUDA versions
    detect_cuda_list
    _build_version_map

    if [ "${#CUDA_MAP[@]}" -eq 0 ]; then
        echo -e "${RED}No CUDA installations found on disk.${NC}"
        return 1
    fi

    # Sort by version (highest last)
    IFS=$'\n' sorted=($(printf '%s\n' "${CUDA_MAP[@]}" | sort -t'|' -k1,1 -V))
    unset IFS
    CUDA_MAP_SORTED=("${sorted[@]}")

    # Keep latest version (highest)
    latest="${CUDA_MAP_SORTED[-1]}"
    latest_ver="${latest%%|*}"
    latest_path="${latest#*|}"

    echo -e "${GREEN}Latest CUDA version will be kept: ${latest_ver} -> ${latest_path}${NC}"

    # Remove all except latest
    for ((i=0; i<${#CUDA_MAP_SORTED[@]}-1; i++)); do
        entry="${CUDA_MAP_SORTED[$i]}"
        ver="${entry%%|*}"
        path="${entry#*|}"

        echo -e "${YELLOW}Removing obsolete CUDA $ver at $path ...${NC}"
        sudo rm -rf "$path"

        # Remove /usr/local/cuda symlink if it points to this version
        if [ -L /usr/local/cuda ]; then
            cur_link=$(readlink -f /usr/local/cuda)
            if [ "$cur_link" = "$path" ]; then
                echo "Removing /usr/local/cuda symlink pointing to removed version"
                sudo rm -f /usr/local/cuda
            fi
        fi
    done

    echo -e "${GREEN}Obsolete CUDA versions removed. Latest version ${latest_ver} retained.${NC}"
}




# ------------------------------
# Main Menu
# ------------------------------
while true; do
    echo -e "\n${GREEN}===== ML/LoRA + RAG Auto Setup Menu =====${NC}"
    check_env

    echo -e "${GREEN}1)${NC} Install NVIDIA drivers"
    echo -e "${GREEN}2)${NC} Install CUDA toolkit"
    echo -e "${GREEN}3)${NC} Install Miniforge (Conda)"
    echo -e "${GREEN}4)${NC} Create new ML environment"
    echo -e "${GREEN}5)${NC} Switch active ML environment"
    echo -e "${GREEN}6)${NC} Validate current ML environment"
    echo -e "${GREEN}7)${NC} Remove an ML environment"
    echo -e "${GREEN}8)${NC} Show disk usage"
    echo -e "${GREEN}9)${NC} Show NVIDIA GPU status (nvidia-smi)"
    echo -e "${GREEN}10)${NC} Show nvcc / CUDA version"
    echo -e "${GREEN}11)${NC} Show Python version in active environment"
    echo -e "${GREEN}12)${NC} Show disk usage of Conda environments"
	echo -e "${GREEN}13)${NC} Detect & select CUDA installation from disk"
	echo -e "${GREEN}14)${NC} Remove CUDA installation(s)"
    echo -e "${GREEN}15)${NC} Remove all obsolete CUDA versions (keep latest)"
	echo -e "${GREEN}0)${NC} Exit / Quit"

    read -p "Choose an option [0-12]: " choice

    case $choice in
        1) install_drivers ;;
        2) install_cuda ;;
        3) install_conda ;;
        4) create_ml_env ;;
        5) switch_env ;;
        6) validate_env ;;
        7) remove_ml_env ;;
        8) show_disk_usage ;;
        9) show_nvidia_smi ;;
        10) show_nvcc_version ;;
        11) show_python_version ;;
        12) env_disk_usage ;;
		13) select_and_persist_cuda ;;
		14) remove_cuda_version ;;
        15) remove_obsolete_cuda_versions ;;
		0) echo "Exiting."; exit 0 ;;
        *) echo -e "${RED}Invalid option. Please enter a number between 0 and 12.${NC}" ;;
    esac
done
