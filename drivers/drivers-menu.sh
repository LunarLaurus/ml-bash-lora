#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
update_script_dir 2

install_drivers() {
    info "${BGREEN}Updating system and installing NVIDIA drivers...${NC}"
    sudo apt update && sudo apt upgrade -y
    sudo ubuntu-drivers devices
    sudo ubuntu-drivers autoinstall
    info "${BGREEN}Drivers installed. Reboot recommended.${NC}"
}

show_nvidia_smi() {
    if ! command -v nvidia-smi &>/dev/null; then
        error "${RED}NVIDIA driver not detected.${NC}"
        return 1
    fi
    nvidia-smi
}

while true; do
    clear
    info "\n${BGREEN}=== NVIDIA Drivers Menu ===${NC}"
    info "1) Install/Update NVIDIA drivers"
    info "2) Show Nvidia-SMI"
    info "${BRED}0) Back to Main Menu${NC}"
    
    read -rp "Choice: " choice
    case $choice in
        1) install_drivers ;;
        2) show_nvidia_smi ;;
        0) break ;;
        *) info "${RED}Invalid option${NC}" ;;
    esac
done
