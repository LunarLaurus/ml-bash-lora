#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
update_script_dir 1

show_disk_usage() {
    echo -e "${GREEN}Disk Usage:${NC}"
    df -h | awk 'NR==1 || /^\/dev\//'
}

while true; do
    echo -e "\n${GREEN}=== Diagnostics Menu ===${NC}"
    echo "1) Show Disk Usage"
    echo "2) Show Python version"
    echo "3) Show Conda environment disk usage"
    echo "0) Back to Main Menu"

    read -rp "Choice: " choice
    case $choice in
        1) show_disk_usage ;;
        2) show_python_version ;;
        3) du -sh $HOME/miniforge/envs/* 2>/dev/null ;;
        0) break ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
