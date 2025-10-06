#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"

# ------------------------------
# Install Miniforge (Conda)
# ------------------------------
install_conda() {
    echo -e "${GREEN}Installing Miniforge3...${NC}"
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh
    bash ~/miniforge.sh -b -p $HOME/miniforge
    eval "$($HOME/miniforge/bin/conda shell.bash hook)"
    conda init
    echo -e "${GREEN}Conda installed. Restart shell to use 'conda activate'.${NC}"
}