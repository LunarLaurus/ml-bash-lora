#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
update_script_dir 2

source "$SCRIPT_DIR/install_conda.sh"
source "$SCRIPT_DIR/env_manager.sh"

# ------------------------------
# Show Disk Usage of Conda Environments
# ------------------------------
env_disk_usage() {
    echo -e "${GREEN}Disk usage of Conda environments:${NC}"
    du -sh $HOME/miniforge/envs/* 2>/dev/null
}

while true; do
    echo -e "\n${GREEN}=== Conda / ML Environment Menu ===${NC}"
    echo "1) Install Miniforge (Conda)"
    echo "2) Create ML environment"
    echo "3) Switch ML environment"
    echo "4) Remove ML environment"
    echo "5) Show environment python version"
    echo "6) Show disk usage for all environments"
    echo "0) Back to Main Menu"

    read -rp "Choice: " choice
    case $choice in
        1) install_conda ;;
        2) create_ml_env ;;
        3) switch_env ;;
        4) remove_ml_env ;;
        5) show_python_version ;;
        6) env_disk_usage ;;
        0) break ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
