#!/bin/bash
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# ------------------------------
# Step 4: Install/ensure PyTorch - uses current env python/pip only
# ------------------------------
install_pytorch_if_missing() {
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
    info "${CYAN}[preflight] Python found: $(python --version 2>/dev/null)${NC}"
    
    ensure_conda || { error "Conda not found. Activate env first."; return 1; }
    info "${CYAN}[preflight] Conda env: ${CONDA_DEFAULT_ENV:-unknown}${NC}"
    
    info "${GREEN}[preflight] Detecting CUDA version...${NC}"
    if ! detect_cuda >/dev/null 2>&1; then
        warn " CUDA not detected; CPU wheel will be used."
    fi
    info "${GREEN}[preflight] CUDA detected: ${CUDA_VER:-none}${NC}"
    
    info "${CYAN}[preflight] Environment check complete.${NC}"
    return 0
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
        error "Torch Index Unset: $TORCH_INDEX_URL"
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
    
    expected_torch_ver="${CUIDX_TO_TORCH_VER[$cuidx]}"
    if [ "$normalized_reported" = "$expected_torch_ver" ]; then
        info "${GREEN}Success: Installed PyTorch matches CUDA $cuda_ver_num.${NC}"
    else
        warn "${RED}Installed PyTorch reports CUDA $normalized_reported, expected $expected_torch_ver.${NC}"
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