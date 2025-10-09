#!/bin/bash


# ------------------------------
# Step 4: Install/ensure PyTorch - uses current env python/pip only
# ------------------------------
install_pytorch_if_missing() {
    info -e "${BLUE}Installing PyTorch stack into current env...${NC}"
    process_deps_lora || { echo -e "${RED}Failed to properly set up and install dependencies!${NC}"; return 1; }
}

process_deps_lora() {
    # Flags to track which steps completed
    local preflight_done=false
    local install_done=false
    local verify_done=false
    
    echo -e "${CYAN}[process] Starting LoRA dependency setup...${NC}"
    
    # --- Preflight ---
    if [[ "$preflight_done" == false ]]; then
        echo -e "${CYAN}[process] Running preflight checks...${NC}"
        preflight_deps || { echo -e "${RED}[process] Preflight checks failed. Aborting.${NC}"; return 1; }
        preflight_done=true
        echo -e "${CYAN}[process] Preflight checks complete.${NC}"
    else
        echo -e "${GREEN}[process] Preflight checks already done, skipping.${NC}"
    fi
    
    # --- Install dependencies ---
    if [[ "$install_done" == false ]]; then
        echo -e "${BLUE}[process] Installing LoRA dependencies...${NC}"
        install_lora_deps || { echo -e "${RED}[process] Dependency installation failed. Aborting.${NC}"; return 1; }
        install_done=true
        echo -e "${BLUE}[process] LoRA dependencies installed.${NC}"
    else
        echo -e "${GREEN}[process] Dependencies already installed, skipping.${NC}"
    fi
    
    # --- Verify ---
    if [[ "$verify_done" == false ]]; then
        echo -e "${BCYAN}[process] Verifying installed dependencies...${NC}"
        verify_deps || { echo -e "${RED}[process] Dependency verification failed. Aborting.${NC}"; return 1; }
        verify_done=true
        echo -e "${BCYAN}[process] Dependency verification complete.${NC}"
    else
        echo -e "${GREEN}[process] Verification already done, skipping.${NC}"
    fi
    
    echo -e "${CYAN}[process] LoRA dependency setup finished successfully.${NC}"
}


preflight_deps() {
    echo -e "${CYAN}[preflight] Checking Python and Conda environment...${NC}"
    
    ensure_python_cmd || { echo -e "${RED}Python not found. Activate env first.${NC}"; return 1; }
    echo -e "${CYAN}[preflight] Python found: $(python --version 2>/dev/null)${NC}"
    
    ensure_conda || { echo -e "${RED}Conda not found. Activate env first.${NC}"; return 1; }
    echo -e "${CYAN}[preflight] Conda env: ${CONDA_DEFAULT_ENV:-unknown}${NC}"
    
    echo -e "${GREEN}[preflight] Detecting CUDA version...${NC}"
    if ! detect_cuda >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning: CUDA not detected; CPU wheel will be used.${NC}"
    fi
    echo -e "${GREEN}[preflight] CUDA detected: ${CUDA_VER:-none}${NC}"
    
    echo -e "${CYAN}[preflight] Environment check complete.${NC}"
    return 0
}

install_lora_deps() {
    echo -e "${BLUE}[install] Upgrading pip/setuptools/wheel...${NC}"
    
    ${PIP_CMD[@]} install --upgrade pip setuptools wheel || return 1
    echo -e "${BLUE}[install] Installing LoRA dependencies from requirements.txt...${NC}"
    
    
    
    if [[ -z "${TORCH_INDEX_URL}" || -z "${CUDA_VER}" ]]; then
        info -e "${YELLOW}Installing PyTorch using CPU wheel index...${NC}"
        ${PIP_CMD[@]} install -r requirements.txt || return 1
    else
        echo -e "${GREEN}[install] Installing PyTorch for CUDA using extra index: ${TORCH_INDEX_URL:-<none>} (CUDA=${CUDA_VER:-unknown})${NC}"
        ${PIP_CMD[@]} install -r requirements.txt --extra-index-url "$TORCH_INDEX_URL" || return 1
    fi
    
    # Validate installed torch
    TORCH_REPORTED="$($PYTHON_CMD -c 'import torch; v=getattr(torch.version,"cuda",None); print(v or "None")')"
    normalized_reported="$(printf '%s' "$TORCH_REPORTED" | sed -E 's/[^0-9.]//g' | grep -oE '^[0-9]+\.[0-9]+')"
    
    expected_torch_ver="${CUIDX_TO_TORCH_VER[$cuidx]}"
    if [ "$normalized_reported" = "$expected_torch_ver" ]; then
        info -e "${GREEN}Success: Installed PyTorch matches CUDA $cuda_ver_num.${NC}"
    else
        warn -e "${RED}Warning: Installed PyTorch reports CUDA $normalized_reported, expected $expected_torch_ver.${NC}"
    fi
    
    echo -e "${BLUE}[install] Dependency installation complete.${NC}"
}


verify_deps() {
    echo -e "${BCYAN}[verify] Verifying installed packages...${NC}"
    echo -e "${BCYAN}[verify] Running import test for torch/numpy...${NC}"
    
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
    
    echo -e "${BCYAN}[verify] Dependency verification complete.${NC}"
}