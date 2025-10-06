#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
update_script_dir 1
source "$SCRIPT_DIR/detect_cuda.sh"

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
