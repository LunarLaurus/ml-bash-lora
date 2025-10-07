#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
update_script_dir 2

install_drivers() {
    echo -e "${BGREEN}Updating system and installing NVIDIA drivers...${NC}"
    sudo apt update && sudo apt upgrade -y
    sudo ubuntu-drivers devices
    sudo ubuntu-drivers autoinstall
    echo -e "${BGREEN}Drivers installed. Reboot recommended.${NC}"
}

show_nvidia_smi() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo -e "${RED}NVIDIA driver not detected.${NC}"
        return 1
    fi
    nvidia-smi
}

while true; do
    echo -e "\n${BGREEN}=== NVIDIA Drivers Menu ===${NC}"
    echo "1) Install/Update NVIDIA drivers"
    echo "2) Show Nvidia-SMI"
    echo -e "${BRED}0) Back to Main Menu${NC}"

    read -rp "Choice: " choice
    case $choice in
        1) install_drivers ;;
        2) show_nvidia_smi ;;
        0) break ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
