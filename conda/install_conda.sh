#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"

# ------------------------------
# Install Miniforge (Conda)
# ------------------------------
install_conda() {
    # Check if conda is already installed
    if command -v conda &>/dev/null; then
        warn "${YELLOW}Conda is already installed:${NC} $(which conda)"
        return 0
    fi
    
    # Check if miniforge folder exists
    if [ -d "$HOME/miniforge" ]; then
        warn "${YELLOW}Miniforge directory already exists at $HOME/miniforge${NC}"
        warn "Attempting to hook Conda into current shell..."
        eval "$($HOME/miniforge/bin/conda shell.bash hook)"
        if command -v conda &>/dev/null; then
            info "${GREEN}Conda hooked successfully.${NC}"
            return 0
        else
            error "${RED}Failed to hook Conda. Consider restarting shell.${NC}"
            return 1
        fi
    fi
    
    # Proceed to download and install Miniforge
    info "${GREEN}Installing Miniforge3...${NC}"
    wget -O ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    
    if [ ! -f ~/miniforge.sh ]; then
        info "${RED}Download failed. Cannot install Conda.${NC}"
        return 1
    fi
    
    bash ~/miniforge.sh -b -p "$HOME/miniforge"
    
    if [ ! -x "$HOME/miniforge/bin/conda" ]; then
        info "${RED}Conda installation failed.${NC}"
        return 1
    fi
    
    # Dynamically hook Conda into current shell
    eval "$($HOME/miniforge/bin/conda shell.bash hook)"
    export PATH="$HOME/miniforge/bin:$PATH"
    
    info "${GREEN}Conda installed and ready to use in this shell session.${NC}"
    info "${YELLOW}You can now run 'conda activate <env>' without restarting your shell.${NC}"
}