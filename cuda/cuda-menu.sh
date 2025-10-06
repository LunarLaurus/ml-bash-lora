#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
update_script_dir
source "$SCRIPT_DIR/detect_cuda.sh"
source "$SCRIPT_DIR/install_cuda.sh"
source "$SCRIPT_DIR/remove_cuda.sh"

while true; do
    echo -e "\n${GREEN}=== CUDA Toolkit Menu ===${NC}"
    echo "1) Detect and select CUDA"
    echo "2) Install CUDA toolkit"
    echo "3) List all CUDA packages from apt-cach"
    echo "4) Remove specific CUDA version(s)"
    echo "5) Remove obsolete CUDA versions (keep latest)"
    echo "6) NVCC Version"
    echo "7) Auto Detect NVCC"
    echo "0) Back to Main Menu"

    read -rp "Choice: " choice
    case $choice in
        1) select_and_persist_cuda ;;
        2) install_cuda ;;
        3) list_available_cuda_versions ;;
        4) remove_cuda_version ;;
        5) remove_obsolete_cuda_versions ;;
        6) show_nvcc_version ;;
        7) auto_detect_nvcc ;;
        0) break ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
