#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/conda/env_manager.sh"
update_script_dir 2
source "$SCRIPT_DIR/git_data.sh"
source "$SCRIPT_DIR/git_utils.sh"
source "$SCRIPT_DIR/python_utils.sh"

# Main menu
while true; do
    echo -e "\n${BGREEN}=== Git Repos Menu (pret: Gen I-IV mainline) ===${NC}"
    echo "1) List repos"
    echo "2) Clone a repo"
    echo "3) Clone ALL repos (skip existing)"
    echo "4) Update (git pull) a repo"
    echo "5) Show git status for a repo"
    echo "6) Delete local repo folder"
    echo "7) Open subshell in repo folder"
    echo -e "\n${CYAN}=== Node/NPM ===${NC}"
    echo "11) Install NVM"
    echo "12) Install NodeJS via NVM"
    echo "13) List NodeJS Versions via NVM"
    echo "14) Use NodeJS Version for NVM"
    echo "15) Set Default NodeJS Version for NVM"
    echo "16) Remove NodeJS Version from NVM"
    echo -e "\n${CYAN}=== LoRA/RAG ===${NC}"
    echo "21) Preflight check"
    echo "22) Install tree-sitter CLI"
    echo "23) Extract code dataset from repo"
    echo "24) Fine-tune LoRA on a repo dataset"
    echo -e "${BRED}0) Back to Main Menu${NC}"
    
    read -rp "Choice: " choice
    case $choice in
        1) list_repos ;;
        2) clone_repo ;;
        3) clone_all_repos ;;
        4) update_repo ;;
        5) repo_status ;;
        6) delete_repo ;;
        7) open_repo_shell ;;
        
        11) install_nvm ;;
        12) install_node ;;
        13) list_node_versions ;;
        14) use_node ;;
        15) set_default_node ;;
        16) remove_node ;;
        
        21) preflight_checks ;;
        22) install_tree_sitter ;;
        23) extract_code_dataset ;;
        24) train_repo_lora ;;
        0) break ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
