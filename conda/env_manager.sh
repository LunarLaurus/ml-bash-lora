#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/cuda/detect_cuda.sh"

# ------------------------------
# ML Environment Functions
# ------------------------------

# Ensure conda is available
ensure_conda() {
    if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniforge/etc/profile.d/conda.sh"
    else
        echo -e "${RED}Conda not found. Install Miniforge first.${NC}"
        return 1
    fi
}

# Show Python Version in Active Env
show_python_version() {
    if [ -z "${PYTHON_CMD:-}" ]; then
        echo -e "${RED}Python not set for active environment.${NC}"
        return 1
    fi
    echo -e "${GREEN}Python version:${NC} $($PYTHON_CMD --version 2>&1)"
}

# Prompt user for environment name and Python version
prompt_env_details() {
    read -rp "Enter environment name [lora]: " ENV_NAME
    ENV_NAME=${ENV_NAME:-lora}
    read -rp "Enter Python version (3.10 or 3.11 recommended) [3.10]: " PY_VER
    PY_VER=${PY_VER:-3.10}
}

# Check if environment exists and handle reinstall/skip
handle_existing_env() {
    if conda env list | grep -qw "$ENV_NAME"; then
        echo -e "${GREEN}Environment $ENV_NAME already exists.${NC}"
        read -rp "Do you want to (R)einstall packages, (S)kip, or (E)xit? [S]: " choice
        choice=${choice:-S}
        case "$choice" in
            R|r) echo "Reinstalling packages..." ;;
            S|s) return 1 ;;  # skip environment creation
            E|e) exit 0 ;;
        esac
    fi
}

# Create and activate environment
create_and_activate_env() {
    # create if missing
    if ! conda env list | grep -qw "$ENV_NAME"; then
        conda create -y -n "$ENV_NAME" python="$PY_VER"
    fi

    # Activate environment in the current shell
    # Use conda activate to set PATH etc.
    conda activate "$ENV_NAME"
    save_env "$ENV_NAME"

    # Set Python/Pip command variables for this env
    PYTHON_CMD="$(which python 2>/dev/null || true)"
    if [ -z "$PYTHON_CMD" ]; then
        # fallback: use conda run to find python path
        PYTHON_CMD="$(conda run -n "$ENV_NAME" which python 2>/dev/null || true)"
    fi
    PIP_CMD="$PYTHON_CMD -m pip"

    echo -e "${GREEN}Activated environment $ENV_NAME.${NC}"
    echo -e "${GREEN}PYTHON_CMD set to: ${PYTHON_CMD}${NC}"
}

# Detect GPU and CUDA (GPU-aware enhancements)
setup_gpu_cuda() {
    # Run existing detection helpers; they may set CUDA_VER / CUDA_PATH
    detect_gpu 2>/dev/null || true
    detect_cuda 2>/dev/null || true

    CUDA_AVAILABLE=0
    # Prefer CUDA_VER set by detect_cuda
    if [ -n "${CUDA_VER:-}" ]; then
        CUDA_AVAILABLE=1
    else
        # fallback checks: nvcc or nvidia-smi presence
        if command -v nvcc &>/dev/null || command -v nvidia-smi &>/dev/null; then
            CUDA_AVAILABLE=1
            # try populate CUDA_VER from nvcc if possible
            if command -v nvcc &>/dev/null; then
                nvcc_ver=$((nvcc --version 2>/dev/null) || true)
                CUDA_VER="$(printf '%s' "$nvcc_ver" | grep -oE 'release [0-9]+\.[0-9]+' | sed 's/release //; s/,//g' || true)"
            else
                # nvidia-smi shows driver-side supported CUDA; not reliable for toolkit version but indicates GPU present
                driver_cuda="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true)"
                CUDA_VER="${driver_cuda:-unknown}"
            fi
        fi
    fi

    if [ "${CUDA_AVAILABLE:-0}" -eq 1 ]; then
        echo -e "${GREEN}CUDA appears available. CUDA_VER='${CUDA_VER:-unknown}'${NC}"
    else
        echo -e "${YELLOW}CUDA not detected.${NC}"
    fi
}

# Install PyTorch if missing (uses existing select_pytorch_wheel logic)
ensure_pytorch() {
    if [ -z "${PYTHON_CMD:-}" ] || [ -z "${PIP_CMD:-}" ]; then
        echo -e "${RED}PYTHON_CMD/PIP_CMD not set. Make sure environment is activated via create_and_activate_env.${NC}"
        return 1
    fi

    if ! $PYTHON_CMD -c "import torch" &>/dev/null; then
        echo -e "${GREEN}PyTorch not found in env. Launching wheel selector...${NC}"
        select_pytorch_wheel
    else
        echo -e "${GREEN}PyTorch already installed.${NC}"
    fi
}

# Install LoRA/Hugging Face stack (GPU-aware where applicable)
install_lora_stack() {
    if [ -z "${PYTHON_CMD:-}" ] || [ -z "${PIP_CMD:-}" ]; then
        echo -e "${RED}PYTHON_CMD/PIP_CMD not set. Ensure create_and_activate_env ran successfully.${NC}"
        return 1
    fi

    local pkgs_common=(transformers peft datasets accelerate)
    for pkg in "${pkgs_common[@]}"; do
        if ! $PYTHON_CMD -c "import ${pkg}" &>/dev/null; then
            echo -e "${GREEN}Installing ${pkg}...${NC}"
            $PIP_CMD install --upgrade "${pkg}" || {
                echo -e "${YELLOW}Failed to install ${pkg} via pip. Consider installing with conda or checking logs.${NC}"
            }
        else
            echo -e "${GREEN}${pkg} already installed.${NC}"
        fi
    done

    # bitsandbytes: attempt pip install; GPU support depends on wheel and system CUDA
    if ! $PYTHON_CMD -c "import bitsandbytes" &>/dev/null; then
        if [ "${CUDA_AVAILABLE:-0}" -eq 1 ]; then
            echo -e "${GREEN}Attempting to install bitsandbytes (GPU-capable if wheel matches CUDA)...${NC}"
        else
            echo -e "${YELLOW}Installing bitsandbytes without detected CUDA; GPU features may not work.${NC}"
        fi
        $PIP_CMD install --upgrade bitsandbytes || {
            echo -e "${RED}bitsandbytes installation failed. To get GPU support you may need matching CUDA toolkit or to build from source. See bitsandbytes docs.${NC}"
        }
    else
        echo -e "${GREEN}bitsandbytes already installed.${NC}"
    fi
}

# Install optional RAG stack (GPU-aware)
install_rag_stack() {
    read -rp "Install RAG stack (faiss, sentence-transformers, langchain)? [y/N]: " rag
    if [[ ! "$rag" =~ ^[Yy]$ ]]; then
        return 0
    fi

    if [ -z "${ENV_NAME:-}" ]; then
        echo -e "${RED}ENV_NAME unknown. Make sure prompt_env_details/create_and_activate_env ran.${NC}"
        return 1
    fi

    # FAISS: prefer GPU conda package if CUDA present
    if [ "${CUDA_AVAILABLE:-0}" -eq 1 ]; then
        echo -e "${GREEN}CUDA detected — attempting to install faiss-gpu via conda (pytorch channel)...${NC}"
        if ! conda install -y -n "$ENV_NAME" -c pytorch faiss-gpu; then
            echo -e "${YELLOW}conda faiss-gpu install failed; falling back to faiss-cpu pip install.${NC}"
            conda run -n "$ENV_NAME" python -m pip install faiss-cpu || {
                echo -e "${RED}Failed to install any FAISS package. Inspect conda/pip output.${NC}"
            }
        fi
    else
        echo -e "${GREEN}Installing faiss-cpu into environment (no CUDA detected)...${NC}"
        conda run -n "$ENV_NAME" python -m pip install faiss-cpu || {
            echo -e "${YELLOW}faiss-cpu pip install failed. Try: conda install -n \"$ENV_NAME\" -c conda-forge faiss-cpu${NC}"
        }
    fi

    # sentence-transformers and langchain: pip installs; they will use GPU if PyTorch is CUDA-enabled
    for pkg in sentence-transformers langchain; do
        if ! conda run -n "$ENV_NAME" python -c "import ${pkg}" &>/dev/null; then
            echo -e "${GREEN}Installing ${pkg} into env ${ENV_NAME}...${NC}"
            conda run -n "$ENV_NAME" python -m pip install "${pkg}" || {
                echo -e "${YELLOW}pip install ${pkg} failed; try manually inside the env.${NC}"
            }
        else
            echo -e "${GREEN}${pkg} already installed in env.${NC}"
        fi
    done

    echo -e "${GREEN}RAG stack installation complete (faiss variant chosen based on CUDA detection).${NC}"
}

# ------------------------------
# Main Create ML Environment Function
# ------------------------------
create_ml_env() {
    ensure_conda || return 1
    prompt_env_details

    if ! handle_existing_env; then
        echo -e "${YELLOW}Skipping environment creation.${NC}"
        return
    fi

    create_and_activate_env

    # create_and_activate_env sets ENV_NAME, PYTHON_CMD, PIP_CMD via conda activate & save_env
    setup_gpu_cuda
    ensure_pytorch
    install_lora_stack
    install_rag_stack
    validate_env
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
            conda deactivate &>/dev/null || true
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
# Validation Function
# ------------------------------
validate_env() {
    check_env || return 1
    # Ensure CURRENT_ENV is defined from ML_ENV_FILE if not set
    if [ -z "${CURRENT_ENV:-}" ] && [ -f "$ML_ENV_FILE" ]; then
        CURRENT_ENV="$(cat "$ML_ENV_FILE")"
    fi

    if [ -z "${CURRENT_ENV:-}" ]; then
        echo -e "${RED}No active environment tracked. Cannot validate.${NC}"
        return 1
    fi

    echo -e "${GREEN}Validating ML environment '${CURRENT_ENV}'...${NC}"

    detect_gpu
    detect_cuda

    # Use conda run to execute checks inside the env to avoid depending on shell activation state
    conda run -n "$CURRENT_ENV" python - <<'PY'
import torch, sys
try:
    if getattr(torch, 'cuda', None) and torch.cuda.is_available():
        print("PyTorch sees GPU:", torch.cuda.get_device_name(0))
    else:
        print("Warning: PyTorch cannot detect GPU")
except Exception as e:
    print("Error checking PyTorch GPU:", e)
PY

    conda run -n "$CURRENT_ENV" python - <<'PY'
try:
    import bitsandbytes as bnb
    print("bitsandbytes imported successfully")
except Exception as e:
    print("bitsandbytes not installed or misconfigured:", e)
PY

    echo -e "${GREEN}Validation complete.${NC}"
}
