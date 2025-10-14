#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
update_script_dir 2
source "$SCRIPT_DIR/detect_cuda.sh"

NVIDIA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"

# ------------------------------
# Ensure NVIDIA CUDA repo exists
# ------------------------------
ensure_nvidia_repo() {
    info "${GREEN}Checking NVIDIA CUDA repository...${NC}"

    # If repo not found in apt sources, add it
    if ! grep -Rq "developer.download.nvidia.com" /etc/apt/sources.list*; then
        info "${YELLOW}NVIDIA CUDA repo not found. Adding it...${NC}"

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
        warn "${YELLOW}⚠ NVIDIA repo is not reachable.${NC}"
        return 1
    fi

    info "${GREEN}NVIDIA repo is valid and reachable.${NC}"
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
    info "\n${GREEN}Detected CUDA versions from apt:${NC}"
    if [ "${#detected_versions[@]}" -eq 0 ]; then
        info "  (none detected via apt-cache)"
    else
        for v in "${detected_versions[@]}"; do
            info "  - $v"
        done
    fi

    # Show default supported versions
    info "\n${GREEN}Default supported CUDA versions:${NC}"
    for s in "${CUDA_SUPPORTED[@]}"; do
        info "  - $s"
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
    info "\n${GREEN}Final available CUDA versions:${NC}"
    if [ "${#versions[@]}" -eq 0 ]; then
        info "  (none available)"
    else
        for v in "${versions[@]}"; do
            info "  - $v"
        done
    fi
}

# ------------------------------
# Suggest and Install CUDA
# ------------------------------
install_cuda() {
    ensure_nvidia_repo || return 1

	list_available_cuda_versions
    info "${GREEN}Suggested CUDA versions for optimal PyTorch compatibility:${NC}"
    info " - 12.9 / 12.8 / 12.6 / 12.4 / 12.3 / 12.2 / 12.1: Recent GPUs & recent PyTorch builds"
    info " - 11.8: Maximum compatibility for older PyTorch releases / older toolchains"
    info "Supported: ${CUDA_SUPPORTED[*]}"
    read -rp "Enter desired CUDA version (major.minor) [12.2]: " CUDA_INPUT
    CUDA_INPUT=${CUDA_INPUT:-12.2}

    # Normalize input (12-2, 12.2.0 -> 12.2)
    CUDA_INPUT="$(printf '%s' "$CUDA_INPUT" | sed -E 's/[-_]//g; s/^([0-9]+)\.([0-9]+).*$/\1.\2/')"

    # Validate version
    if ! [[ " ${CUDA_SUPPORTED[*]} " =~ " ${CUDA_INPUT} " ]]; then
        warn "${YELLOW}Requested version '$CUDA_INPUT' not in supported list. Defaulting to 12.2.${NC}"
        CUDA_INPUT="12.2"
    fi

    pkg_ver="${CUDA_INPUT//./-}"
    cuda_pkg="cuda-${pkg_ver}"

    info "${GREEN}Installing CUDA package: $cuda_pkg${NC}"
    sudo apt-get install -y "$cuda_pkg" || {
        warn "${YELLOW}Initial install failed; retrying with -f install...${NC}"
        sudo apt-get -f install -y
        sudo apt-get install -y "$cuda_pkg" || {
            error "${RED}Failed to install $cuda_pkg.${NC}"
            return 1
        }
    }
    update_torch_index_url

    info "${GREEN}CUDA $CUDA_INPUT installed successfully.${NC}"
	info "${GREEN}Run detect_cuda to select and persist this installation.${NC}"

}
