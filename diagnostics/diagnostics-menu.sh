#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
update_script_dir 2

show_disk_usage() {
    info "${GREEN}Disk Usage:${NC}"
    df -h | awk 'NR==1 || /^\/dev\//'
}

while true; do
    info "\n${YELLOW}=== Diagnostics Menu ===${NC}"
    info "1) Show Disk Usage"
    info "${BRED}0) Back to Main Menu${NC}"
    
    read -rp "Choice: " choice
    case $choice in
        1) show_disk_usage ;;
        0) break ;;
        *) info "${RED}Invalid option${NC}" ;;
    esac
done
