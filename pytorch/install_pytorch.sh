#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/cuda/detect_cuda.sh"
update_script_dir 2

# ------------------------------
# Select and install PyTorch wheel (auto-installs & validates CUDA <-> torch match)
# ------------------------------
select_pytorch_wheel() {
    # Ensure CUDA is detected and persisted if missing
    detect_cuda --persist || true

    # Use the Python from the active conda env
    PYTHON_CMD=$(which python)
    PIP_CMD="$PYTHON_CMD -m pip"

    # Get CUDA version (major.minor)
    cuda_ver_num="${CUDA_VER:-}"
    if [ -n "$cuda_ver_num" ]; then
        cuda_ver_num=$(printf '%s' "$cuda_ver_num" | grep -oE '^[0-9]+\.[0-9]+')
    fi

    # If no CUDA, default to CPU wheel
    if [ -z "$cuda_ver_num" ]; then
        echo -e "${YELLOW}No CUDA detected. Installing CPU-only PyTorch wheel.${NC}"
        SUGGESTED="torch torchvision torchaudio"
        echo -e "${GREEN}Installing: $SUGGESTED${NC}"
        $PIP_CMD install --upgrade $SUGGESTED || return 1
        echo -e "${GREEN}CPU PyTorch installed.${NC}"
        return 0
    fi

    # CUDA version -> PyTorch cu index
    declare -A CUDA_TO_CUIDX=(
        ["11.8"]="cu118"
        ["12.1"]="cu121"
        ["12.2"]="cu122"
        ["12.3"]="cu123"
        ["12.4"]="cu124"
        ["12.6"]="cu126"
        ["12.8"]="cu128"
        ["12.9"]="cu129"
    )
    declare -A CUIDX_TO_TORCH_VER
    for k in "${!CUDA_TO_CUIDX[@]}"; do
        CUIDX_TO_TORCH_VER[${CUDA_TO_CUIDX[$k]}]=$k
    done

    cuidx="${CUDA_TO_CUIDX[$cuda_ver_num]}"
    if [ -z "$cuidx" ]; then
        echo -e "${YELLOW}No mapping for detected CUDA $cuda_ver_num. Defaulting to CPU wheel.${NC}"
        SUGGESTED="torch torchvision torchaudio"
        $PIP_CMD install --upgrade $SUGGESTED || return 1
        return 0
    fi

    SUGGESTED="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${cuidx}"
    echo -e "${GREEN}Installing PyTorch for CUDA $cuda_ver_num using wheel index $cuidx...${NC}"
    $PIP_CMD install --upgrade $SUGGESTED || return 1

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