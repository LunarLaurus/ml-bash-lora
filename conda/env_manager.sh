#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/cuda/detect_cuda.sh"


PYTHON_CMD=""
PIP_CMD=()

# Detect Python version
PY_VER=""

# ------------------------------
# Helpers: ensure PYTHON_CMD/PIP_CMD point to current (activated) env
# ------------------------------
ensure_python_cmd() {
    # Already set and valid?
    if [ -n "${PYTHON_CMD:-}" ] && [ -x "$PYTHON_CMD" ]; then
        if ! command -v "$PYTHON_CMD" &>/dev/null; then
            echo -e "${YELLOW}Warning: PYTHON_CMD exists but not executable. Resetting.${NC}"
            PYTHON_CMD=""
            PIP_CMD=()
        else
            
            PY_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            info "Detected Python version: $PY_VER"
            return 0
        fi
    fi
    
    # Try ML_ENV_FILE
    if [ -f "$ML_ENV_FILE" ]; then
        get_active_env || true
        candidate="$HOME/miniforge/envs/${CURRENT_ENV}/bin/python"
        if [ -x "$candidate" ]; then
            PYTHON_CMD="$candidate"
            
            PY_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            info "Detected Python version: $PY_VER"
        else
            echo -e "${YELLOW}Python not found at $candidate${NC}"
        fi
    fi
    
    # Python on PATH
    if [ -z "${PYTHON_CMD:-}" ] && command -v python &>/dev/null; then
        PYTHON_CMD="$(command -v python)"
        
        PY_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        info "Detected Python version: $PY_VER"
    fi
    
    # Check final python
    if [ -z "${PYTHON_CMD:-}" ] || [ ! -x "$PYTHON_CMD" ]; then
        echo -e "${RED}No valid Python found in current shell or env.${NC}"
        PYTHON_CMD=""
        PIP_CMD=()
        return 1
    fi
    
    # Set pip as array to avoid quoting issues
    PIP_CMD=("$PYTHON_CMD" "-m" "pip")
    
    # Ensure pip is installed
    if ! "${PIP_CMD[@]}" --version &>/dev/null; then
        echo -e "${YELLOW}pip not found; bootstrapping pip in this Python...${NC}"
        "$PYTHON_CMD" -m ensurepip --upgrade || {
            echo -e "${RED}Failed to bootstrap pip${NC}"
            PIP_CMD=()
            return 1
        }
    fi
    
    return 0
}

# Run Python inline code in the current environment
run_python_inline() {
    ensure_python_cmd || { echo -e "${RED}Python not found for active environment.${NC}"; return 1; }
    
    "$PYTHON_CMD" - <<'PYCODE'
# You can put any Python code here
# Example: print("Hello from Python!")
PYCODE
}

# Run a Python script in the current environment with arguments
run_python_file() {
    ensure_python_cmd || { echo -e "${RED}Python not found for active environment.${NC}"; return 1; }
    
    local script="$1"; shift
    if [ ! -f "$script" ]; then
        echo -e "${RED}Python script not found: $script${NC}"
        return 1
    fi
    
    "$PYTHON_CMD" "$script" "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo -e "${RED}Python script '$script' failed (exit $rc)${NC}"
    fi
    return $rc
}


# ------------------------------
# Preserve original logic: ensure conda available only when needed
# (but most functions will not call conda; they use current activated env's python/pip)
# ------------------------------
ensure_conda() {
    if command -v conda &>/dev/null; then
        return 0
    fi
    
    # Hook conda into current shell if available
    if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        source "$HOME/miniforge/etc/profile.d/conda.sh"
        export PATH="$HOME/miniforge/bin:$PATH"
        return 0
    fi
    
    if [ -x "$HOME/miniforge/bin/conda" ]; then
        # shellcheck disable=SC1090
        source "$HOME/miniforge/etc/profile.d/conda.sh"
        export PATH="$HOME/miniforge/bin:$PATH"
        return 0
    fi
    
    echo -e "${YELLOW}Conda not found in PATH. Some operations (create env, conda faiss-gpu) will be unavailable.${NC}"
    return 1
}

get_active_env() {
    if [ -f "$ML_ENV_FILE" ]; then
        CURRENT_ENV="$(cat "$ML_ENV_FILE")"
        if [ -z "$CURRENT_ENV" ]; then
            return 1
        fi
        return 0
    else
        return 1
    fi
}

save_env() {
    echo "$1" > "$ML_ENV_FILE"
}

# ------------------------------
# Step 1: Prompt details / Create env
# Note: create_env will try to use conda if available, otherwise tell user how to create env.
# ------------------------------
prompt_env_details() {
    read -rp "Enter environment name [lora]: " ENV_NAME
    ENV_NAME=${ENV_NAME:-lora}
    read -rp "Enter Python version (3.10 or 3.11 recommended) [3.10]: " PY_VER
    PY_VER=${PY_VER:-3.10}
}

handle_existing_env() {
    # If env exists on disk (miniforge envs) or conda knows about it, ask user
    exists=0
    if [ -d "$HOME/miniforge/envs/$ENV_NAME" ]; then
        exists=1
        elif command -v conda &>/dev/null && conda env list | grep -qw "$ENV_NAME"; then
        exists=1
    fi
    
    if [ "$exists" -eq 1 ]; then
        echo -e "${GREEN}Environment $ENV_NAME already exists.${NC}"
        read -rp "Do you want to (R)einstall packages, (S)kip creation, or (E)xit? [S]: " choice
        choice=${choice:-S}
        case "$choice" in
            R|r)
                echo "Reinstalling curated packages..."
                reinstall_packages
                return 0
            ;;
            S|s) return 1 ;;  # skip environment creation
            E|e) exit 0 ;;
        esac
    fi
}

create_env() {
    if [ -z "${ENV_NAME:-}" ]; then
        prompt_env_details
    fi
    
    # If conda available, attempt to create
    if command -v conda &>/dev/null; then
        if conda env list | grep -qw "$ENV_NAME"; then
            echo -e "${YELLOW}Environment '$ENV_NAME' already exists (conda).${NC}"
            return 0
        fi
        echo -e "${GREEN}Creating conda env '$ENV_NAME' (python $PY_VER)...${NC}"
        conda create -y -n "$ENV_NAME" python="$PY_VER" || {
            echo -e "${RED}Failed to create environment with conda.${NC}"
            return 1
        }
        echo -e "${GREEN}Environment '$ENV_NAME' created.${NC}"
        return 0
    fi
    
    # No conda: try to create directory placeholder and instruct user
    echo -e "${YELLOW}Conda not available. Please create the environment manually (e.g. locally or with conda on another shell).${NC}"
    echo "Suggested command (run when conda is available):"
    echo "  conda create -y -n $ENV_NAME python=$PY_VER"
    return 1
}

# ------------------------------
# Step 2: 'Activate' / set env variables
# Activation in-script should NOT call conda; instead we use current shell activation or ML_ENV_FILE
# ------------------------------
activate_env() {
    # If conda available and user wants activation in the current shell, we can call conda activate
    if [ -z "${ENV_NAME:-}" ]; then
        read -rp "Enter environment name to activate: " ENV_NAME
        ENV_NAME=${ENV_NAME:-lora}
    fi
    
    # If conda command exists, prefer to activate to set PATH for this shell
    if command -v conda &>/dev/null; then
        ensure_conda >/dev/null 2>&1
        conda activate "$ENV_NAME" 2>/dev/null || {
            echo -e "${YELLOW}conda activate failed in this shell; ensure conda init has been run.${NC}"
        }
    else
        # No conda command: user should have already activated env in this shell by other means
        echo -e "${YELLOW}Conda command unavailable. This script will rely on the currently active Python on PATH.${NC}"
    fi
    
    # Save the tracked env name for menus / later use
    save_env "$ENV_NAME"
    
    # Set PYTHON_CMD/PIP_CMD based on current shell python (assumes env activated)
    if ! ensure_python_cmd; then
        echo -e "${YELLOW}Could not determine python for environment $ENV_NAME. Ensure it is activated in this shell or that $HOME/miniforge/envs/$ENV_NAME/bin/python exists.${NC}"
    else
        echo -e "${GREEN}Environment '$ENV_NAME' set. PYTHON_CMD=${PYTHON_CMD}${NC}"
    fi
}

# ------------------------------
# Step 3: GPU/CUDA detection (unchanged)
# ------------------------------
setup_gpu_cuda() {
    detect_gpu 2>/dev/null || true
    detect_cuda 2>/dev/null || true
    set_cuda_available
}

set_cuda_available() {
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
# Step 4: Install/ensure PyTorch - uses current env python/pip only
# ------------------------------
install_pytorch_if_missing() {
    echo -e "${BLUE}Installing PyTorch stack into current env...${NC}"
    ensure_python_cmd || { echo -e "${RED}Python not found in active env. Activate env first.${NC}"; return 1; }
    
    if ! "$PYTHON_CMD" -c "import torch" &>/dev/null; then
        echo -e "${GREEN}PyTorch not found in env. Launching wheel selector (this will use the active env)...${NC}"
        # select_pytorch_wheel should use PYTHON_CMD/PIP_CMD; ensure it does not call conda
        select_pytorch_wheel
    else
        echo -e "${GREEN}PyTorch already installed in env.${NC}"
    fi
}

# ------------------------------
# Step 5: Install LoRA stack - pip into current env
# ------------------------------
install_lora_stack() {
    echo -e "${BLUE}Installing LoRA stack into current env...${NC}"
    ensure_python_cmd || { echo -e "${RED}Python not found. Activate env first.${NC}"; return 1; }
    ensure_conda || { echo -e "${RED}Conda not found. Activate env first.${NC}"; return 1; }
    detect_cuda >/dev/null 2>&1 || echo -e "${YELLOW}Warning: CUDA not detected; CPU wheel will be used.${NC}"
    set_cuda_available
    
    local pkgs=(transformers peft datasets accelerate)
    for pkg in "${pkgs[@]}"; do
        if ! "$PYTHON_CMD" -c "import ${pkg}" &>/dev/null; then
            echo -e "${GREEN}Installing ${pkg} into current env...${NC}"
            "${PIP_CMD[@]}" install --upgrade "${pkg}" || echo -e "${YELLOW}Failed to install ${pkg} via pip.${NC}"
        else
            echo -e "${GREEN}${pkg} already installed in current env.${NC}"
        fi
    done
    
    # bitsandbytes via pip (GPU support depends on wheel / system CUDA)
    if ! "$PYTHON_CMD" -c "import bitsandbytes" &>/dev/null; then
        # Get numeric CUDA index (default 0 if none/mapping missing)
        cuda_version_number=$(get_cuda_version_index)
        
        if [ "${CUDA_AVAILABLE:-0}" -eq 1 ] && [ "$cuda_version_number" -ne 0 ]; then
            echo -e "${GREEN}Attempting to install bitsandbytes (GPU-capable wheel if available)...${NC}"
            "${PIP_CMD[@]}" install --upgrade "bitsandbytes-cuda$cuda_version_number" || \
            echo -e "${RED}bitsandbytes install failed.${NC}"
        else
            echo -e "${YELLOW}Installing bitsandbytes without detected CUDA; GPU features may not work.${NC}"
            "${PIP_CMD[@]}" install --upgrade bitsandbytes || \
            echo -e "${RED}bitsandbytes install failed.${NC}"
        fi
    else
        echo -e "${GREEN}bitsandbytes already installed in current env.${NC}"
    fi
    
    
}

# ------------------------------
# Step 6: Install RAG stack - prefer GPU faiss only when conda available; else fall back to pip CPU
# ------------------------------
install_rag_stack() {
    echo -e "${BLUE}Installing RAG stack into current env...${NC}"
    ensure_python_cmd || { echo -e "${RED}Python not found. Activate env first.${NC}"; return 1; }
    ensure_conda || { echo -e "${RED}Conda not found. Activate env first.${NC}"; return 1; }
    detect_cuda >/dev/null 2>&1 || echo -e "${YELLOW}Warning: CUDA not detected; CPU wheel will be used.${NC}"
    set_cuda_available
    
    read -rp "Install RAG stack (faiss, sentence-transformers, langchain)? [y/N]: " rag
    if [[ ! "$rag" =~ ^[Yy]$ ]]; then
        return 0
    fi
    
    # FAISS: prefer faiss-gpu via conda if conda exists and user has CUDA; otherwise use pip cpu option
    if [ "${CUDA_AVAILABLE:-0}" -eq 1 ] && command -v conda &>/dev/null; then
        echo -e "${GREEN}Conda available and CUDA detected.${NC}"
        echo -e "${GREEN}Attempting: conda install -y -n <active env> -c pytorch faiss-gpu${NC}"
        # If conda is present but we must not call it directly (user said avoid), we still try to call only if available.
        if ! conda install -y -n "${ENV_NAME:-$(cat $ML_ENV_FILE 2>/dev/null || echo '')}" -c pytorch faiss-gpu; then
            echo -e "${YELLOW}conda faiss-gpu failed or not possible. Falling back to pip faiss-cpu.${NC}"
            "${PIP_CMD[@]}" install faiss-cpu || echo -e "${RED}faiss-cpu pip install failed.${NC}"
        fi
    else
        echo -e "${GREEN}Installing faiss-cpu into current env (no conda/CUDA combo detected)...${NC}"
        "${PIP_CMD[@]}" install faiss-cpu || echo -e "${YELLOW}faiss-cpu pip install failed.${NC}"
    fi
    
    # sentence-transformers and langchain via pip into current env
    for pkg in sentence-transformers langchain; do
        if ! "$PYTHON_CMD" -c "import ${pkg}" &>/dev/null; then
            echo -e "${GREEN}Installing ${pkg} into current env...${NC}"
            "${PIP_CMD[@]}" install "${pkg}" || echo -e "${YELLOW}Failed to pip install ${pkg}.${NC}"
        else
            echo -e "${GREEN}${pkg} already installed in current env.${NC}"
        fi
    done
}

# ------------------------------
# Step 7: Validate environment using the current env python (no conda run)
# ------------------------------
validate_env() {
    # Determine active env name from ML_ENV_FILE (best-effort)
    if [ -z "${ENV_NAME:-}" ] && [ -f "$ML_ENV_FILE" ]; then
        ENV_NAME="$(cat "$ML_ENV_FILE")"
    fi
    
    echo -e "${GREEN}Validating environment '${ENV_NAME:-(unknown)}' using current python...${NC}"
    
    detect_gpu || true
    detect_cuda || true
    set_cuda_available
    
    ensure_python_cmd || { echo -e "${RED}Python not found in current shell. Activate env first.${NC}"; return 1; }
    ensure_conda || { echo -e "${RED}Conda not found. Activate env first.${NC}"; return 1; }
    
    "$PYTHON_CMD" - <<'PY'
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
# Reinstall curated packages (when user chooses R on existing env)
# ------------------------------
reinstall_packages() {
    # Use current env python/pip; if not set, attempt to set
    ensure_python_cmd || { echo -e "${RED}Python for current env not found. Activate env first.${NC}"; return 1; }
    
    echo -e "${GREEN}Reinstalling curated LoRA packages into current env...${NC}"
    # Call the same install functions which are idempotent
    install_pytorch_if_missing || true
    install_lora_stack || true
    # Ask about RAG reinstall
    read -rp "Also reinstall RAG stack? [y/N]: " do_rag
    if [[ "$do_rag" =~ ^[Yy]$ ]]; then
        install_rag_stack || true
    fi
    echo -e "${GREEN}Reinstall finished.${NC}"
}

# ------------------------------
# Switch env: do not call conda if unavailable. Save to ML_ENV_FILE and instruct user how to activate.
# ------------------------------
switch_env() {
    read -rp "Enter environment name to switch to: " NEW_ENV
    if [ -z "$NEW_ENV" ]; then
        echo -e "${YELLOW}No env given.${NC}"
        return 1
    fi
    
    ensure_conda
    
    # If conda exists, check env presence via conda; otherwise check miniforge env dir
    if command -v conda &>/dev/null; then
        if ! conda env list | grep -qw "$NEW_ENV"; then
            echo -e "${RED}Environment '$NEW_ENV' not found via conda.${NC}"
            return 1
        fi
    else
        if [ ! -d "$HOME/miniforge/envs/$NEW_ENV" ]; then
            echo -e "${RED}Environment directory not found at $HOME/miniforge/envs/$NEW_ENV and conda not available.${NC}"
            return 1
        fi
    fi
    
    save_env "$NEW_ENV"
    ensure_python_cmd || echo -e "${YELLOW}Warning: PYTHON_CMD/PIP_CMD may not be valid after switching.${NC}"
    echo -e "${GREEN}Active environment updated to '$NEW_ENV' (saved to $ML_ENV_FILE).${NC}"
    if command -v conda &>/dev/null; then
        echo -e "${GREEN}Run: conda activate $NEW_ENV${NC} to actually activate it in this shell."
    else
        echo -e "${YELLOW}No conda in PATH: activate the environment by starting a shell where it is available or ensure $HOME/miniforge/envs/$NEW_ENV/bin is in PATH.${NC}"
    fi
    
    # attempt to update PYTHON_CMD/PIP_CMD based on saved env
    PYTHON_CMD="$HOME/miniforge/envs/$NEW_ENV/bin/python"
    if [ -x "$PYTHON_CMD" ]; then
        PIP_CMD="$PYTHON_CMD -m pip"
        echo -e "${GREEN}PYTHON_CMD updated to $PYTHON_CMD (if you activated env elsewhere this will match).${NC}"
    fi
}

# ------------------------------
# Show Python Version in Active Env (conda-neutral)
# ------------------------------
show_python_version() {
    ensure_python_cmd || { echo -e "${RED}Python not found in current shell/environment.${NC}"; return 1; }
    echo -e "${GREEN}Python executable:${NC} $PYTHON_CMD"
    "$PYTHON_CMD" --version 2>&1
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


# ------------------------------
# Select and install PyTorch wheel (auto-installs & validates CUDA <-> torch match)
# ------------------------------
select_pytorch_wheel() {
    # Ensure CUDA is detected and persisted if missing
    detect_cuda >/dev/null 2>&1 || echo -e "${YELLOW}Warning: CUDA not detected; CPU wheel will be used.${NC}"
    
    
    # Use the Python from the active conda env
    # PYTHON_CMD=$(which python)
    # PIP_CMD="$PYTHON_CMD -m pip"
    ensure_python_cmd
    
    # Get PyTorch CUDA index
    cuidx=$(get_cu_index)
    if [ -z "$cuidx" ]; then
        echo -e "${YELLOW}No mapping for detected CUDA $cuda_ver_num. Defaulting to CPU wheel.${NC}"
        SUGGESTED="torch torchvision torchaudio"
        ${PIP_CMD[@]} install --upgrade $SUGGESTED || return 1
        return 0
    fi
    
    SUGGESTED="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${cuidx}"
    echo -e "${GREEN}Installing PyTorch for CUDA $cuda_ver_num using wheel index $cuidx...${NC}"
    ${PIP_CMD[@]} install --upgrade $SUGGESTED || return 1
    
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
# Convenience: run full setup using the steps above (uses current env; create_env may require conda)
# ------------------------------
run_full_setup() {
    create_env || { echo -e "${YELLOW}create_env failed or skipped.${NC}"; }
    activate_env || { echo -e "${YELLOW}activate_env failed.${NC}"; }
    setup_gpu_cuda || true
    install_pytorch_if_missing || true
    install_lora_stack || true
    install_rag_stack || true
    validate_env || true
}
