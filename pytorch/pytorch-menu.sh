#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
update_script_dir 2
source "$SCRIPT_DIR/install_pytorch.sh"

while true; do
    echo -e "\n${GREEN}=== PyTorch Installation Menu ===${NC}"
    echo "1) Install/Upgrade PyTorch for current CUDA"
    echo "0) Back to Main Menu"

    read -rp "Choice: " choice
    case $choice in
        1) select_pytorch_wheel ;;
        0) break ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
