#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/conda/conda-menu.sh"
update_script_dir 2

ensure_python_cmd || { info -e "${RED}Python not found for active environment. Activate/Create an Env. first.${NC}"; }
select_project
CURRENT_REPO_PATH=""

update_script_dir() {
    CURRENT_REPO_PATH="$1"
}

current_repo_path_check() {
    if [ -z "$CURRENT_REPO_PATH" ]; then
        echo "No repository selected. Please select a repository first."
        return 1
    fi
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install dependencies for the current repo
install_dependencies() {
    if current_repo_path_check; then
        ensure_python_cmd || { echo -e "${RED}Python not found. Activate env first.${NC}"; return 1; }
        
        check_python_deps numpy torch torchvision torchaudio transformers datasets peft torch tqdm numpy scipy sklearn tiktoken google.protobuf bitsandbytes accelerate safetensors
        if [ ${#MISSING_PY_DEPS[@]} -gt 0 ]; then
            warn "${BRED}Missing Python dependencies: ${MISSING_PY_DEPS[*]}${NC}"
            auto_install_python_deps || {
                error "${BRED}Automatic installation failed. Please install manually:${NC} pip install ${MISSING_PY_DEPS[*]}"
                return 1
            }
            info "${BGREEN}Dependencies installed successfully.${NC}"
        fi
    fi
}

# Function to run a script with the current repo context
run_script() {
    SCRIPT_PATH="$1"
    shift
    if current_repo_path_check; then
        "$PYTHON_CMD" "$SCRIPT_PATH" "$CURRENT_REPO_PATH" "$@"
    fi
}

select_project(){
    if ! prompt_repo_selection; then
        info "Cancelled"
        return 0
    fi
    resolve_selection_to_folder "$REPO_SEL" || return 1
    ensure_python_cmd || { error "Python not found. Activate env first."; return 1; }
    # compute FOLDER_PATH if index selected
    if [[ "$REPO_SEL" =~ ^[0-9]+$ ]]; then
        FOLDER_PATH="$(repo_folder_from_url "${poke_repos_mainline[$REPO_SEL]}")"
    fi
    if [ -z "${FOLDER_PATH:-}" ]; then
        FOLDER_PATH="$REPO_SEL"
    fi
}

# Main menu loop
while true; do
    clear
    echo "[qwen2.5-coder:7b] Lora System Script Menu"
    echo "-------------------------"
    echo "1. 01_index_files.py"
    echo "2. 02_parse_code.py"
    echo "3. 02b_build_dependency_graphs.py"
    echo "4. 03_enrich_with_model.py"
    echo "5. 04_link_headers.py"
    echo "6. 05_generate_qna.py"
    echo "7. 06_train_lora.py"
    echo "8. 07_embed_code.py"
    echo "9. 08_query_system.py"
    echo "10. 09_evaluate.py"
    echo "11. Install Dependencies"
    echo "12. Exit"
    read -p "Enter your choice [1-12]: " choice
    
    case $choice in
        1) update_script_dir "$CURRENT_REPO_PATH"; run_script 01_index_files.py "$@" ;;
        2) update_script_dir "$CURRENT_REPO_PATH"; run_script 02_parse_code.py "$@" ;;
        3) update_script_dir "$CURRENT_REPO_PATH"; run_script 02b_build_dependency_graphs.py "$@" ;;
        4) update_script_dir "$CURRENT_REPO_PATH"; run_script 03_enrich_with_model.py "$@" ;;
        5) update_script_dir "$CURRENT_REPO_PATH"; run_script 04_link_headers.py "$@" ;;
        6) update_script_dir "$CURRENT_REPO_PATH"; run_script 05_generate_qna.py "$@" ;;
        7) update_script_dir "$CURRENT_REPO_PATH"; run_script 06_train_lora.py "$@" ;;
        8) update_script_dir "$CURRENT_REPO_PATH"; run_script 07_embed_code.py "$@" ;;
        9) update_script_dir "$CURRENT_REPO_PATH"; run_script 08_query_system.py "$@" ;;
        10) update_script_dir "$CURRENT_REPO_PATH"; run_script 09_evaluate.py "$@" ;;
        11) install_dependencies ;;
        12) exit 0 ;;
        *) echo "Invalid choice. Please try again." ;;
    esac
    
    read -p "Press Enter to continue..."
done