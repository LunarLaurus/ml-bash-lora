# env_manager.sh
#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/cuda/detect_cuda.sh"

# ------------------------------
# ML Environment Functions
# ------------------------------

# Ensure conda is available
ensure_conda() {
    if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        source "$HOME/miniforge/etc/profile.d/conda.sh"
        return 0
    else
        echo -e "${RED}Conda not found. Install Miniforge first.${NC}"
        return 1
    fi
}

# ------------------------------
# Step 1: Prompt details / Create env
# ------------------------------
prompt_env_details() {
    read -rp "Enter environment name [lora]: " ENV_NAME
    ENV_NAME=${ENV_NAME:-lora}
    read -rp "Enter Python version (3.10 or 3.11 recommended) [3.10]: " PY_VER
    PY_VER=${PY_VER:-3.10}
}

create_env() {
    ensure_conda || return 1
    if [ -z "${ENV_NAME:-}" ]; then
        prompt_env_details
    fi

    if conda env list | grep -qw "$ENV_NAME"; then
        echo -e "${YELLOW}Environment '$ENV_NAME' already exists.${NC}"
        return 0
    fi

    echo -e "${GREEN}Creating conda env '$ENV_NAME' (python $PY_VER)...${NC}"
    conda create -y -n "$ENV_NAME" python="$PY_VER" || {
        echo -e "${RED}Failed to create environment.$NC"
        return 1
    }
    echo -e "${GREEN}Environment '$ENV_NAME' created.${NC}"
}

# ------------------------------
# Step 2: Activate / set env variables
# ------------------------------
activate_env() {
    ensure_conda || return 1
    if [ -z "${ENV_NAME:-}" ]; then
        read -rp "Enter environment name to activate: " ENV_NAME
        ENV_NAME=${ENV_NAME:-lora}
    fi

    if ! conda env list | grep -qw "$ENV_NAME"; then
        echo -e "${RED}Environment '$ENV_NAME' does not exist. Create it first.${NC}"
        return 1
    fi

    # activate in current shell
    conda activate "$ENV_NAME"
    save_env "$ENV_NAME"

    # set PYTHON_CMD/PIP_CMD
    PYTHON_CMD="$(which python 2>/dev/null || true)"
    if [ -z "$PYTHON_CMD" ]; then
        PYTHON_CMD="$(conda run -n "$ENV_NAME" which python 2>/dev/null || true)"
    fi
    PIP_CMD="$PYTHON_CMD -m pip"

    echo -e "${GREEN}Activated environment '$ENV_NAME'.${NC}"
    echo -e "${GREEN}PYTHON_CMD=${PYTHON_CMD}${NC}"
}

# ------------------------------
# Step 3: GPU/CUDA detection
# ------------------------------
setup_gpu_cuda() {
    # Run detect helpers (they may set CUDA_VER/CUDA_PATH)
    detect_gpu 2>/dev/null || true
    detect_cuda 2>/dev/null || true

    CUDA_AVAILABLE=0
    if [ -n "${CUDA_VER:-}" ]; then
        CUDA_AVAILABLE=1
    else
        if command -v nvcc &>/dev/null || command -v nvidia-smi &>/dev/null; then
            CUDA_AVAILABLE=1
            if command -v nvcc &>/dev/null; then
                nvcc_ver="$(nvcc --version 2>/dev/null || true)"
                CUDA_VER="$(printf '%s' "$nvcc_ver" | grep -oE 'release [0-9]+\.[0-9]+' | sed 's/release //; s/,//g' || true)"
            fi
        fi
    fi

    if [ "${CUDA_AVAILABLE:-0}" -eq 1 ]; then
        echo -e "${GREEN}CUDA detected. CUDA_VER='${CUDA_VER:-unknown}'${NC}"
    else
        echo -e "${YELLOW}CUDA not detected.${NC}"
    fi
}

# ------------------------------
# Step 4: Install/ensure PyTorch (uses select_pytorch_wheel helper)
# ------------------------------
install_pytorch_if_missing() {
    if [ -z "${PYTHON_CMD:-}" ]; then
        echo -e "${RED}PYTHON_CMD not set. Activate the environment first.${NC}"
        return 1
    fi
    if ! $PYTHON_CMD -c "import torch" &>/dev/null; then
        echo -e "${GREEN}PyTorch not found, launching wheel selector...${NC}"
        select_pytorch_wheel
    else
        echo -e "${GREEN}PyTorch already installed in env.${NC}"
    fi
}

# ------------------------------
# Step 5: Install LoRA stack
# ------------------------------
install_lora_stack() {
    if [ -z "${PYTHON_CMD:-}" ]; then
        echo -e "${RED}PYTHON_CMD not set. Activate environment first.${NC}"
        return 1
    fi

    local pkgs=(transformers peft datasets accelerate)
    for pkg in "${pkgs[@]}"; do
        if ! $PYTHON_CMD -c "import ${pkg}" &>/dev/null; then
            echo -e "${GREEN}Installing ${pkg}...${NC}"
            $PIP_CMD install --upgrade "${pkg}" || echo -e "${YELLOW}Failed to install ${pkg} (pip).${NC}"
        else
            echo -e "${GREEN}${pkg} already installed.${NC}"
        fi
    done

    # bitsandbytes (GPU-capable if wheel matches CUDA)
    if ! $PYTHON_CMD -c "import bitsandbytes" &>/dev/null; then
        if [ "${CUDA_AVAILABLE:-0}" -eq 1 ]; then
            echo -e "${GREEN}Installing bitsandbytes (attempt GPU-capable wheel)...${NC}"
        else
            echo -e "${YELLOW}Installing bitsandbytes without detected CUDA (GPU features may not work).${NC}"
        fi
        $PIP_CMD install --upgrade bitsandbytes || echo -e "${RED}bitsandbytes install failed.${NC}"
    else
        echo -e "${GREEN}bitsandbytes already installed.${NC}"
    fi
}

# ------------------------------
# Step 6: Install RAG stack (GPU-aware faiss)
# ------------------------------
install_rag_stack() {
    if [ -z "${PYTHON_CMD:-}" ]; then
        echo -e "${RED}PYTHON_CMD not set. Activate environment first.${NC}"
        return 1
    fi

    read -rp "Install RAG stack (faiss, sentence-transformers, langchain)? [y/N]: " rag
    if [[ ! "$rag" =~ ^[Yy]$ ]]; then
        return 0
    fi

    # FAISS: choose GPU or CPU
    if [ "${CUDA_AVAILABLE:-0}" -eq 1 ]; then
        echo -e "${GREEN}Attempting to install faiss-gpu via conda (pytorch channel)...${NC}"
        if ! conda install -y -n "$ENV_NAME" -c pytorch faiss-gpu; then
            echo -e "${YELLOW}faiss-gpu install failed; falling back to faiss-cpu via pip.${NC}"
            conda run -n "$ENV_NAME" python -m pip install faiss-cpu || echo -e "${RED}faiss install failed.${NC}"
        fi
    else
        echo -e "${GREEN}Installing faiss-cpu (no CUDA detected)...${NC}"
        conda run -n "$ENV_NAME" python -m pip install faiss-cpu || echo -e "${YELLOW}faiss-cpu pip install failed.${NC}"
    fi

    # sentence-transformers and langchain
    for pkg in sentence-transformers langchain; do
        if ! conda run -n "$ENV_NAME" python -c "import ${pkg}" &>/dev/null; then
            echo -e "${GREEN}Installing ${pkg}...${NC}"
            conda run -n "$ENV_NAME" python -m pip install "${pkg}" || echo -e "${YELLOW}Failed to pip install ${pkg}.${NC}"
        else
            echo -e "${GREEN}${pkg} already installed in env.${NC}"
        fi
    done
}

# ------------------------------
# Step 7: Validate environment
# ------------------------------
validate_env() {
    check_env || return 1

    # determine active env
    if [ -z "${CURRENT_ENV:-}" ] && [ -f "$ML_ENV_FILE" ]; then
        CURRENT_ENV="$(cat "$ML_ENV_FILE")"
    fi
    if [ -z "${CURRENT_ENV:-}" ]; then
        echo -e "${RED}No active environment tracked. Cannot validate.${NC}"
        return 1
    fi

    echo -e "${GREEN}Validating ML environment '${CURRENT_ENV}'...${NC}"

    # GPU and CUDA info (host)
    detect_gpu || true
    detect_cuda || true

    # python checks inside env
    conda run -n "$CURRENT_ENV" python - <<'PY'
import sys
try:
    import torch
    if getattr(torch, 'cuda', None) and torch.cuda.is_available():
        print("PyTorch sees GPU:", torch.cuda.get_device_name(0))
    else:
        print("Warning: PyTorch cannot detect GPU")
except Exception as e:
    print("Error checking PyTorch:", e)

try:
    import bitsandbytes as bnb
    print("bitsandbytes imported OK")
except Exception as e:
    print("bitsandbytes: NOT OK or not installed:", e)
PY

    echo -e "${GREEN}Validation complete.${NC}"
}

# ------------------------------
# Convenience: run full setup as ordered steps
# ------------------------------
run_full_setup() {
    # run steps in order, move on if a step fails but report
    create_env || { echo -e "${YELLOW}create_env failed or skipped.${NC}"; }
    activate_env || { echo -e "${YELLOW}activate_env failed.${NC}"; }
    setup_gpu_cuda || true
    install_pytorch_if_missing || true
    install_lora_stack || true
    install_rag_stack || true
    validate_env || true
}
