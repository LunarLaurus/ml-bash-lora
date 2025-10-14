#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"

echo "SCRIPT_DIR: $SCRIPT_DIR"
while true; do
    echo -e "\n${GREEN}===== ML/LoRA + RAG Main Menu =====${NC}"
    check_env || true
    
    echo -e "${BGREEN}1) NVIDIA Drivers${NC}"
    echo -e "${BGREEN}2) CUDA Toolkit${NC}"
    echo -e "${BCYAN}3) Conda / ML Environments${NC}"
    echo -e "${YELLOW}4) Git/Pokemon${NC}"
    echo -e "${BMAGENTA}5) Qwen${NC}"
    echo -e "${YELLOW}6) Diagnostics${NC}"
    echo -e "${CYAN}7) Utilities${NC}"
    echo -e "${BRED}0) Exit${NC}"
    
    read -rp "Choice: " choice
    case $choice in
        1) source ./drivers/drivers-menu.sh ;;
        2) source ./cuda/cuda-menu.sh ;;
        3) source ./conda/conda-menu.sh ;;
        4) source ./git/git-menu.sh ;;
        5) source ./qwen/qwen-menu.sh ;;
        6) source ./diagnostics/diagnostics-menu.sh ;;
        7) source ./utils/utils-menu.sh ;;
        0) echo "Exiting."; exit 0 ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
