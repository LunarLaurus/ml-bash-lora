#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"

# ------------------------------
# Install Miniforge (Conda)
# ------------------------------
install_conda() {
    # Check if conda is already installed
    if command -v conda &>/dev/null; then
        echo -e "${YELLOW}Conda is already installed:${NC} $(which conda)"
        return 0
    fi

    # Check if miniforge folder exists
    if [ -d "$HOME/miniforge" ]; then
        echo -e "${YELLOW}Miniforge directory already exists at $HOME/miniforge${NC}"
        echo "Attempting to hook Conda into current shell..."
        eval "$($HOME/miniforge/bin/conda shell.bash hook)"
        if command -v conda &>/dev/null; then
            echo -e "${GREEN}Conda hooked successfully.${NC}"
            return 0
        else
            echo -e "${RED}Failed to hook Conda. Consider restarting shell.${NC}"
            return 1
        fi
    fi

    # Proceed to download and install Miniforge
    echo -e "${GREEN}Installing Miniforge3...${NC}"
    wget -O ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

    if [ ! -f ~/miniforge.sh ]; then
        echo -e "${RED}Download failed. Cannot install Conda.${NC}"
        return 1
    fi

    bash ~/miniforge.sh -b -p "$HOME/miniforge"

    if [ ! -x "$HOME/miniforge/bin/conda" ]; then
        echo -e "${RED}Conda installation failed.${NC}"
        return 1
    fi

    # Dynamically hook Conda into current shell
    eval "$($HOME/miniforge/bin/conda shell.bash hook)"
    export PATH="$HOME/miniforge/bin:$PATH"

    echo -e "${GREEN}Conda installed and ready to use in this shell session.${NC}"
    echo -e "${YELLOW}You can now run 'conda activate <env>' without restarting your shell.${NC}"
}