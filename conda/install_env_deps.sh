#!/bin/bash
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# CUDA → Recommended PyTorch version mapping (Ubuntu, Python 3.10/3.11)
# Validated for CUDA 11.x–12.9 and current PyTorch wheel availability
declare -A CUIDX_TO_TORCH_VER=(
    ["cu118"]="2.7"   # CUDA 11.8 → stable baseline (broadest support)
    ["cu121"]="2.8"   # CUDA 12.1 → supported since 2.8
    ["cu122"]="2.8"   # CUDA 12.2 → supported since 2.8
    ["cu123"]="2.8"   # CUDA 12.3 → supported since 2.8
    ["cu124"]="2.8"   # CUDA 12.4 → supported since 2.8
    ["cu126"]="2.8"   # CUDA 12.6 → supported since 2.8
    ["cu128"]="2.8"   # CUDA 12.8 → current stable (preferred for 2025 GPUs)
    ["cu129"]="2.8"   # CUDA 12.9 → latest available index, 2.8 wheels exist
    ["cu0"]=""        # CPU-only / unknown CUDA → fallback to CPU wheels
)

# ------------------------------
# Step 4: Install/ensure PyTorch - uses current env python/pip only
# ------------------------------
install_pytorch_if_missing() {
    check_env || { error "Error"; return 1; }
    info "${BBLUE}Installing PyTorch stack into current env...${NC}"
    process_deps_lora || { error "Failed to properly set up and install dependencies!"; return 1; }
}

process_deps_lora() {
    # Flags to track which steps completed
    local preflight_done=false
    local install_done=false
    local verify_done=false
    
    info "${CYAN}[process] Starting LoRA dependency setup...${NC}"
    
    # --- Preflight ---
    if [[ "$preflight_done" == false ]]; then
        info "${CYAN}[process] Running preflight checks...${NC}"
        preflight_deps || { error "[process] Preflight checks failed. Aborting."; return 1; }
        preflight_done=true
        info "${CYAN}[process] Preflight checks complete.${NC}"
    else
        info "${GREEN}[process] Preflight checks already done, skipping.${NC}"
    fi
    
    # --- Install dependencies ---
    if [[ "$install_done" == false ]]; then
        info "${BBLUE}[process] Installing LoRA dependencies...${NC}"
        install_lora_deps || { error "[process] Dependency installation failed. Aborting."; return 1; }
        install_done=true
        info "${BBLUE}[process] LoRA dependencies installed.${NC}"
    else
        info "${GREEN}[process] Dependencies already installed, skipping.${NC}"
    fi
    
    # --- Verify ---
    if [[ "$verify_done" == false ]]; then
        info "${BCYAN}[process] Verifying installed dependencies...${NC}"
        verify_deps || { error "[process] Dependency verification failed. Aborting."; return 1; }
        verify_done=true
        info "${BCYAN}[process] Dependency verification complete.${NC}"
    else
        info "${GREEN}[process] Verification already done, skipping.${NC}"
    fi
    
    info "${CYAN}[process] LoRA dependency setup finished successfully.${NC}"
}


preflight_deps() {
    info "${CYAN}[preflight] Checking Python and Conda environment...${NC}"
    
    ensure_python_cmd || { error "Python not found. Activate env first."; return 1; }
    info "${CYAN}[preflight] Python found: $($PYTHON_CMD --version 2>/dev/null)${NC}"
    
    ensure_conda || { error "Conda not found. Activate env first."; return 1; }
    info "${CYAN}[preflight] Conda env: ${CONDA_DEFAULT_ENV:-unknown}${NC}"
    
    info "${BGREEN}[preflight] Detecting CUDA version...${NC}"
    if ! detect_cuda >/dev/null 2>&1; then
        warn " CUDA not detected; CPU wheel will be used."
    fi
    info "${BGREEN}[preflight] CUDA detected: ${CUDA_VER:-none}${NC}"
    
    info "${CYAN}[preflight] Environment check complete.${NC}"
    return 0
}

# Helper: return expected torch version for a given cuidx.
# - normalises the cuidx
# - falls back to "cu0" (CPU) for empty/invalid keys
# - echoes the version (may be empty string for CPU)
get_expected_torch_ver() {
    local cuidx="$1"
    # Default to cu0 if empty/null
    if [[ -z "${cuidx:-}" ]]; then
        cuidx="cu0"
    fi
    
    # Normalize to lower-case (just in case)
    cuidx="${cuidx,,}"
    
    # If key exists in the associative array, return its value
    if [[ -v "CUIDX_TO_TORCH_VER[$cuidx]" ]]; then
        printf '%s' "${CUIDX_TO_TORCH_VER[$cuidx]}"
        return 0
    fi
    
    # Unknown cuidx -> fallback to cu0
    printf '%s' "${CUIDX_TO_TORCH_VER[cu0]}"
    return 0
}

list_installed_packages() {
    info "Listing installed Python packages (pip list):"
    "${PIP_CMD[@]}" list || warn "Failed to list packages with pip list"
    
    info "Listing installed Python packages (pip freeze):"
    "${PIP_CMD[@]}" freeze || warn "Failed to list packages with pip freeze"
}


remove_user_packages() {
    read -rp "Enter package names to remove (space-separated): " -a user_pkgs
    
    if [[ ${#user_pkgs[@]} -eq 0 ]]; then
        info "No packages entered. Nothing to do."
        return 0
    fi
    
    for pkg in "${user_pkgs[@]}"; do
        info "Removing $pkg ..."
        "${PIP_CMD[@]}" uninstall -y "$pkg" || warn "Failed to uninstall $pkg"
    done
}

install_lora_deps() {
    info "${BBLUE}[install] Upgrading pip/setuptools/wheel...${NC}"
    
    ${PIP_CMD[@]} install --upgrade pip wheel setuptools || return 1
    info "${BBLUE}[install] Installing LoRA dependencies from requirements.txt...${NC}"
    
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        error "Cannot find requirements.txt at $REQUIREMENTS_FILE"
        return 1
    fi
    
    if [[ -z "${TORCH_INDEX_URL}" ]]; then
        error "Torch Index not set, updating."
        update_torch_index_url
    fi
    
    if [[ -z "${TORCH_INDEX_URL}" || -z "${CUDA_VER}" ]]; then
        warn "Installing PyTorch using CPU wheel index..."
        ${PIP_CMD[@]} install -r "$REQUIREMENTS_FILE" || return 1
    else
        info "Installing PyTorch for CUDA using extra index: ${TORCH_INDEX_URL:-<none>} (CUDA=${CUDA_VER:-unknown})"
        ${PIP_CMD[@]} install -r "$REQUIREMENTS_FILE" --extra-index-url "$TORCH_INDEX_URL" || return 1
    fi
    
    # Validate installed torch
    TORCH_REPORTED="$($PYTHON_CMD -c 'import torch; v=getattr(torch.version,"cuda",None); print(v or "None")')"
    normalized_reported="$(printf '%s' "$TORCH_REPORTED" | sed -E 's/[^0-9.]//g' | grep -oE '^[0-9]+\.[0-9]+')"
    cuidx="$(get_cu_index)"
    expected_torch_ver="$(get_expected_torch_ver "$cuidx")"
    expected_label="${expected_torch_ver:-cpu}"
    
    local installed_torch_ver
    installed_torch_ver="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
    
    if [[ "${installed_torch_ver%%.*}" = "${expected_torch_ver%%.*}" ]]; then
        info "${GREEN}Success: Installed PyTorch ${installed_torch_ver} matches expected ${expected_torch_ver}.${NC}"
    else
        warn "${RED}Mismatch: Installed PyTorch ${installed_torch_ver}, expected ${expected_torch_ver}.${NC}"
    fi
    
    local reported_cuda_ver
    reported_cuda_ver="$(python -c 'import torch; print(torch.version.cuda or "cpu")')"
    
    local expected_cuda="${CUDA_VER:-}"
    
    if [[ "$reported_cuda_ver" == "cpu" && "$expected_cuda" == "cpu" ]]; then
        info "${GREEN}Success: CPU-only PyTorch installation.${NC}"
        elif [[ "${reported_cuda_ver/./}" == "${expected_cuda/./}" ]]; then
        info "${GREEN}Success: PyTorch CUDA runtime ($reported_cuda_ver) matches expected ($expected_cuda).${NC}"
    else
        warn "${RED}Mismatch: PyTorch built with CUDA $reported_cuda_ver, expected $expected_cuda for $cuidx.${NC}"
    fi
    
    info "${BBLUE}[install] Dependency installation complete.${NC}"
}

verify_deps() {
    info "${BCYAN}[verify] Verifying installed packages...${NC}"
    info "${BCYAN}[verify] Running import test for torch/numpy...${NC}"
    
    run_python_inline <<'PYCODE'
import importlib, sys
try:
    torch = importlib.import_module("torch")
    numpy = importlib.import_module("numpy")
    print(f"Torch: {getattr(torch, '__version__', 'unknown')}")
    try:
        print(f"CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print("CUDA available: error checking:", e)
    print(f"NumPy: {getattr(numpy, '__version__', 'unknown')}")
except Exception as e:
    print("Error importing dependencies:", e)
    sys.exit(1)
PYCODE
    
    info "${BCYAN}[verify] Dependency verification complete.${NC}"
}