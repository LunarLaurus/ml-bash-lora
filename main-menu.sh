#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"

echo "SCRIPT_DIR: $SCRIPT_DIR"
while true; do
    echo -e "\n${GREEN}===== ML/LoRA + RAG Main Menu =====${NC}"
    check_env || true

    echo -e "1) NVIDIA Drivers"
    echo -e "2) CUDA Toolkit"
    echo -e "3) Conda / ML Environments"
    echo -e "4) PyTorch Installation"
    echo -e "5) Diagnostics"
    echo -e "6) Utilities"
    echo -e "0) Exit"

    read -rp "Choice: " choice
    case $choice in
        1) source ./drivers/drivers-menu.sh ;;
        2) source ./cuda/cuda-menu.sh ;;
        3) source ./conda/conda-menu.sh ;;
        4) source ./pytorch/pytorch-menu.sh ;;
        5) source ./diagnostics/diagnostics-menu.sh ;;
        5) source ./utils/utils-menu.sh ;;
        0) echo "Exiting."; exit 0 ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
