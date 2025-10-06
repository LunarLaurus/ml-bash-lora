#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"

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
    conda create -y -n "$ENV_NAME" python="$PY_VER"
    conda activate "$ENV_NAME"
    save_env "$ENV_NAME"
    PYTHON_CMD=$(which python)
    PIP_CMD="$PYTHON_CMD -m pip"
    echo -e "${GREEN}Activated environment $ENV_NAME.${NC}"
}

# Detect GPU and CUDA
setup_gpu_cuda() {
    detect_gpu
    detect_cuda
}

# Install PyTorch if missing
ensure_pytorch() {
    if ! $PYTHON_CMD -c "import torch" &>/dev/null; then
        select_pytorch_wheel
    else
        echo -e "${GREEN}PyTorch already installed.${NC}"
    fi
}

# Install LoRA/Hugging Face stack
install_lora_stack() {
    local packages=(transformers peft datasets accelerate bitsandbytes)
    for pkg in "${packages[@]}"; do
        if ! $PYTHON_CMD -c "import $pkg" &>/dev/null; then
            echo -e "${GREEN}Installing $pkg...${NC}"
            $PIP_CMD install --upgrade "$pkg"
        else
            echo -e "${GREEN}$pkg already installed.${NC}"
        fi
    done
}

# Install optional RAG stack
install_rag_stack() {
    read -rp "Install RAG stack (faiss, sentence-transformers, langchain)? [y/N]: " rag
    if [[ "$rag" =~ ^[Yy]$ ]]; then
        local packages=(faiss-cpu sentence-transformers langchain)
        for pkg in "${packages[@]}"; do
            if ! $PYTHON_CMD -c "import $pkg" &>/dev/null; then
                echo -e "${GREEN}Installing $pkg...${NC}"
                $PIP_CMD install "$pkg"
            else
                echo -e "${GREEN}$pkg already installed.${NC}"
            fi
        done
    fi
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