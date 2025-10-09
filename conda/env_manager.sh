#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/cuda/detect_cuda.sh"
source "$PROJECT_ROOT/conda/install_env_deps.sh"

PY_VER=""
PYTHON_CMD=""
CONDA_DEFAULT_ENV=""
PIP_CMD=()


ensure_requirements(){
    ensure_conda || { error "Conda not found. Activate env first."; return 1; }
    detect_cuda >/dev/null 2>&1 || warn "Warning: CUDA not detected; CPU wheel will be used."
    set_cuda_available
}

# ------------------------------
# Helpers: ensure PYTHON_CMD/PIP_CMD point to current (activated) env
# ------------------------------
ensure_python_cmd() {
    # Already set and valid?
    if [ -n "${PYTHON_CMD:-}" ] && [ -x "$PYTHON_CMD" ]; then
        if ! command -v "$PYTHON_CMD" &>/dev/null; then
            warn "Warning: PYTHON_CMD exists but not executable. Resetting."
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
            warn "Python not found at $candidate"
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
        error "No valid Python found in current shell or env."
        PYTHON_CMD=""
        PIP_CMD=()
        return 1
    fi
    
    # Set pip as array to avoid quoting issues
    PIP_CMD=("$PYTHON_CMD" "-m" "pip")
    
    # Ensure pip is installed
    if ! "${PIP_CMD[@]}" --version &>/dev/null; then
        warn "pip not found; bootstrapping pip in this Python..."
        "$PYTHON_CMD" -m ensurepip --upgrade || {
            error "Failed to bootstrap pip"
            PIP_CMD=()
            return 1
        }
    fi
    
    return 0
}

# Run Python inline code in the current environment
run_python_inline() {
    ensure_conda || { error "Problem with environment, read above."; return 1; }
    "$PYTHON_CMD" - <<'PYCODE'
# You can put any Python code here
# Example: print("Hello from Python!")
PYCODE
}

# Run a Python script in the current environment with arguments
run_python_file() {
    ensure_conda || { error "Problem with environment, read above."; return 1; }
    local script="$1"; shift
    if [ ! -f "$script" ]; then
        error "Python script not found: $script"
        return 1
    fi
    
    "$PYTHON_CMD" "$script" "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        error "Python script '$script' failed (exit $rc)"
    fi
    return $rc
}


# ------------------------------
# Preserve original logic: ensure conda available only when needed
# (but most functions will not call conda; they use current activated env's python/pip)
# ------------------------------
ensure_conda() {
    ensure_python_cmd || { error "Python not found for active environment."; return 1; }
    if command -v conda &>/dev/null; then
        export CONDA_DEFAULT_ENV="$(conda info --base >/dev/null && conda info --json | jq -r '.active_prefix_name' 2>/dev/null || echo "")"
        info "[DEBUG] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>} (returning from ${FUNCNAME[1]})"
        return 0
    fi
    
    # Hook conda into current shell if available
    if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        source "$HOME/miniforge/etc/profile.d/conda.sh"
        export PATH="$HOME/miniforge/bin:$PATH"
        export CONDA_DEFAULT_ENV="$(conda info --json | jq -r '.active_prefix_name' 2>/dev/null || echo "")"
        info "[DEBUG] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>} (returning from ${FUNCNAME[1]})"
        return 0
    fi
    
    if [ -x "$HOME/miniforge/bin/conda" ]; then
        # shellcheck disable=SC1090
        source "$HOME/miniforge/etc/profile.d/conda.sh"
        export PATH="$HOME/miniforge/bin:$PATH"
        export CONDA_DEFAULT_ENV="$(conda info --json | jq -r '.active_prefix_name' 2>/dev/null || echo "")"
        info "[DEBUG] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>} (returning from ${FUNCNAME[1]})"
        return 0
    fi
    
    warn "Conda not found in PATH. Some operations (create env, conda faiss-gpu) will be unavailable."
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
    read -rp "Enter environment name [CHANGE_ME]: " ENV_NAME
    ENV_NAME=${ENV_NAME:-CHANGE_ME}
    read -rp "Enter Python version (3.10 or 3.11 recommended) [3.11]: " PY_VER
    PY_VER=${PY_VER:-3.11}
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
        info "${GREEN}Environment $ENV_NAME already exists.${NC}"
        read -rp "Do you want to (R)einstall packages, (S)kip creation, or (E)xit? [S]: " choice
        choice=${choice:-S}
        case "$choice" in
            R|r)
                info "Reinstalling curated packages..."
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
    ensure_conda || { error "Problem with environment, read above."; return 1; }
    
    # If conda available, attempt to create
    if command -v conda &>/dev/null; then
        if conda env list | grep -qw "$ENV_NAME"; then
            warn "Environment '$ENV_NAME' already exists (conda)."
            activate_env
            return 0
        fi
        info "${GREEN}Creating conda env '$ENV_NAME' (python $PY_VER)...${NC}"
        conda create -y -n "$ENV_NAME" python="$PY_VER" || {
            error "Failed to create environment with conda."
            return 1
        }
        info "${GREEN}Environment '$ENV_NAME' created.${NC}"
        activate_env
        return 0
    fi
    
    # No conda: try to create directory placeholder and instruct user
    warn "Conda not available. Please create the environment manually (e.g. locally or with conda on another shell)."
    warn "Suggested command (run when conda is available):"
    warn "  conda create -y -n $ENV_NAME python=$PY_VER"
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
    ensure_conda || { error "Problem with environment, read above."; return 1; }
    
    # Save the tracked env name for menus / later use
    save_env "$ENV_NAME"
    
    # Set PYTHON_CMD/PIP_CMD based on current shell python (assumes env activated)
    if ! ensure_python_cmd; then
        warn "Could not determine python for environment $ENV_NAME. Ensure it is activated in this shell or that $HOME/miniforge/envs/$ENV_NAME/bin/python exists."
    else
        info "${GREEN}Environment '$ENV_NAME' set. PYTHON_CMD=${PYTHON_CMD}${NC}"
    fi
}

# ------------------------------
# Step 3: GPU/CUDA detection (unchanged)
# ------------------------------
setup_gpu_cuda() {
    detect_gpu 2>/dev/null || true
    ensure_requirements
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
        info "${GREEN}CUDA detected. CUDA_VER='${CUDA_VER:-unknown}'${NC}"
    else
        warn "CUDA not detected."
    fi
}

# ------------------------------
# Step 5: Install LoRA stack - pip into current env
# ------------------------------
install_lora_stack() {
    info "${BLUE}Installing LoRA stack into current env...${NC}"
    ensure_requirements || { error "Problem with environment, read above."; return 1; }
    local pkgs=(transformers peft datasets accelerate)
    for pkg in "${pkgs[@]}"; do
        if ! "$PYTHON_CMD" -c "import ${pkg}" &>/dev/null; then
            info "${GREEN}Installing ${pkg} into current env...${NC}"
            "${PIP_CMD[@]}" install --upgrade "${pkg}" || warn "Failed to install ${pkg} via pip."
        else
            info "${GREEN}${pkg} already installed in current env.${NC}"
        fi
    done
    
    # bitsandbytes via pip (GPU support depends on wheel / system CUDA)
    if ! "$PYTHON_CMD" -c "import bitsandbytes" &>/dev/null; then
        # Get numeric CUDA index (default 0 if none/mapping missing)
        cuda_version_number=$(get_cuda_version_index)
        
        if [ "${CUDA_AVAILABLE:-0}" -eq 1 ] && [ "$cuda_version_number" -ne 0 ]; then
            info "${GREEN}Attempting to install bitsandbytes (GPU-capable wheel if available)...${NC}"
            "${PIP_CMD[@]}" install --upgrade "bitsandbytes-cuda$cuda_version_number" || \
            error "bitsandbytes install failed."
        else
            info "${YELLOW}Installing bitsandbytes without detected CUDA; GPU features may not work.${NC}"
            "${PIP_CMD[@]}" install --upgrade bitsandbytes || \
            error "bitsandbytes install failed."
        fi
    else
        info "${GREEN}bitsandbytes already installed in current env.${NC}"
    fi
    
    
}

# ------------------------------
# Step 6: Install RAG stack - prefer GPU faiss only when conda available; else fall back to pip CPU
# ------------------------------
install_rag_stack() {
    info "${BLUE}Installing RAG stack into current env...${NC}"
    ensure_requirements || { error "Problem with environment, read above."; return 1; }
    read -rp "Install RAG stack (faiss, sentence-transformers, langchain)? [y/N]: " rag
    if [[ ! "$rag" =~ ^[Yy]$ ]]; then
        return 0
    fi
    
    # FAISS: prefer faiss-gpu via conda if conda exists and user has CUDA; otherwise use pip cpu option
    if [ "${CUDA_AVAILABLE:-0}" -eq 1 ] && command -v conda &>/dev/null; then
        info "${GREEN}Conda available and CUDA detected.${NC}"
        info "${GREEN}Attempting: conda install -y -n <active env> -c pytorch faiss-gpu${NC}"
        # If conda is present but we must not call it directly (user said avoid), we still try to call only if available.
        if ! conda install -y -n "${ENV_NAME:-$(cat $ML_ENV_FILE 2>/dev/null || echo '')}" -c pytorch faiss-gpu; then
            warn "conda faiss-gpu failed or not possible. Falling back to pip faiss-cpu."
            "${PIP_CMD[@]}" install faiss-cpu || error "faiss-cpu pip install failed."
        fi
    else
        warn "Installing faiss-cpu into current env (no conda/CUDA combo detected)..."
        "${PIP_CMD[@]}" install faiss-cpu || error "faiss-cpu pip install failed."
    fi
    
    # sentence-transformers and langchain via pip into current env
    for pkg in sentence-transformers langchain; do
        if ! "$PYTHON_CMD" -c "import ${pkg}" &>/dev/null; then
            info "${GREEN}Installing ${pkg} into current env...${NC}"
            "${PIP_CMD[@]}" install "${pkg}" || warn "Failed to pip install ${pkg}."
        else
            info "${GREEN}${pkg} already installed in current env.${NC}"
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
    
    echo "${GREEN}Validating environment '${ENV_NAME:-(unknown)}' using current python...${NC}"
    setup_gpu_cuda
    
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
    
    info "${GREEN}Validation complete.${NC}"
}

# ------------------------------
# Reinstall curated packages (when user chooses R on existing env)
# ------------------------------
reinstall_packages() {
    # Use current env python/pip; if not set, attempt to set
    ensure_conda || { error "Problem with environment, read above."; return 1; }
    
    info "${GREEN}Reinstalling curated LoRA packages into current env...${NC}"
    # Call the same install functions which are idempotent
    install_pytorch_if_missing || true
    install_lora_stack || true
    # Ask about RAG reinstall
    read -rp "Also reinstall RAG stack? [y/N]: " do_rag
    if [[ "$do_rag" =~ ^[Yy]$ ]]; then
        install_rag_stack || true
    fi
    info "${GREEN}Reinstall finished.${NC}"
}

# ------------------------------
# Switch env: do not call conda if unavailable. Save to ML_ENV_FILE and instruct user how to activate.
# ------------------------------
switch_env() {
    read -rp "Enter environment name to switch to: " NEW_ENV
    if [ -z "$NEW_ENV" ]; then
        warn "No env given."
        return 1
    fi
    
    
    ensure_conda || { error "Problem with environment, read above."; return 1; }
    
    # If conda exists, check env presence via conda; otherwise check miniforge env dir
    if command -v conda &>/dev/null; then
        if ! conda env list | grep -qw "$NEW_ENV"; then
            warn "Environment '$NEW_ENV' not found via conda."
            return 1
        fi
    else
        if [ ! -d "$HOME/miniforge/envs/$NEW_ENV" ]; then
            error "Environment directory not found at $HOME/miniforge/envs/$NEW_ENV and conda not available."
            return 1
        fi
    fi
    
    save_env "$NEW_ENV"
    ensure_conda || { error "PYTHON_CMD/PIP_CMD may not be valid after switching."; return 1; }
    info "${GREEN}Active environment updated to '$NEW_ENV' (saved to $ML_ENV_FILE).${NC}"
    if command -v conda &>/dev/null; then
        info "${GREEN}Run: conda activate $NEW_ENV to actually activate it in this shell.${NC}"
    else
        warn "No conda in PATH: activate the environment by starting a shell where it is available or ensure $HOME/miniforge/envs/$NEW_ENV/bin is in PATH."
    fi
    
    # attempt to update PYTHON_CMD/PIP_CMD based on saved env
    PYTHON_CMD="$HOME/miniforge/envs/$NEW_ENV/bin/python"
    if [ -x "$PYTHON_CMD" ]; then
        PIP_CMD="$PYTHON_CMD -m pip"
        info "${GREEN}PYTHON_CMD updated to $PYTHON_CMD (if you activated env elsewhere this will match).${NC}"
    fi
}

# ------------------------------
# Remove ML Environment / Project
# ------------------------------
remove_ml_env() {
    ensure_conda || { error "Problem with environment, read above."; return 1; }
    conda env list
    read -p "Enter the environment name to remove: " ENV_NAME
    if conda env list | grep -qw "$ENV_NAME"; then
        read -p "Are you sure you want to permanently delete '$ENV_NAME'? [y/N]: " CONFIRM
        if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
            conda deactivate &>/dev/null || true
            conda env remove -n "$ENV_NAME"
            info "${GREEN}Environment '$ENV_NAME' removed successfully.${NC}"
            
            # Clear active environment if it was the removed one
            if [ -f "$ML_ENV_FILE" ] && grep -qw "$ENV_NAME" "$ML_ENV_FILE"; then
                rm -f "$ML_ENV_FILE"
                info "${GREEN}Active environment cleared.${NC}"
            fi
        else
            info "Aborted environment removal."
        fi
    else
        warn "Environment '$ENV_NAME' does not exist."
    fi
}

# ------------------------------
# Show Python Version in Active Env (conda-neutral)
# ------------------------------
show_python_version() {
    ensure_conda || { error "Problem with environment, read above."; return 1; }
    info "${GREEN}Python executable:${NC} $PYTHON_CMD"
    "$PYTHON_CMD" --version 2>&1
}

# ------------------------------
# Convenience: run full setup using the steps above (uses current env; create_env may require conda)
# ------------------------------
run_full_setup() {
    create_env || { warn "create_env failed or skipped."; }
    activate_env || { warn "activate_env failed."; }
    setup_gpu_cuda || true
    install_pytorch_if_missing || true
    install_lora_stack || true
    install_rag_stack || true
    validate_env || true
}
