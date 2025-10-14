#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/conda/env_manager.sh"

CURRENT_REPO_PATH=""

current_repo_path_check() {
    if [ -z "$CURRENT_REPO_PATH" ]; then
        echo "No repository selected. Please select a repository first."
        return 1
    fi
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
    select_project
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
    resolve_selection_to_folder "$REPO_CHOICE" || return 1
    ensure_python_cmd || { error "Python not found. Activate env first."; return 1; }
    # compute FOLDER_PATH if index selected
    if [[ "$REPO_CHOICE" =~ ^[0-9]+$ ]]; then
        CURRENT_REPO_PATH="$(repo_folder_from_url "${poke_repos_mainline[$REPO_CHOICE]}")"
    fi
    if [ -z "${CURRENT_REPO_PATH:-}" ]; then
        CURRENT_REPO_PATH="$REPO_CHOICE"
    fi
}

ensure_python_cmd || { info -e "${RED}Python not found for active environment. Activate/Create an Env. first.${NC}"; }
# Main menu loop
while true; do
    clear
    echo "[qwen2.5-coder:7b] Lora System Script Menu"
    echo "-------------------------"
    echo "1) Index Project Files "
    echo "2) Parse Indexed Code"
    echo "3) Build Dependency Graph"
    echo "4) Enrich"
    echo "5) Link Headers"
    echo "6) Generate QNA"
    echo "7) Train LoRA"
    echo "8) Embed Code"
    echo "9) Query System"
    echo "10) Evaluate"
    echo "21) Install Dependencies"
    echo "0) Exit"
    read -p "Enter your choice [1-12]: " choice
    
    case $choice in
        1) run_script 01_index_files.py ;;
        2) run_script 02_parse_code.py ;;
        3) run_script 02b_build_dependency_graphs.py ;;
        4) run_script 03_enrich_with_model.py ;;
        5) run_script 04_link_headers.py ;;
        6) run_script 05_generate_qna.py ;;
        7) run_script 06_train_lora.py ;;
        8) run_script 07_embed_code.py ;;
        9) run_script 08_query_system.py ;;
        10) run_script 09_evaluate.py ;;
        21) install_dependencies ;;
        0) exit 0 ;;
        *) echo "Invalid choice. Please try again." ;;
    esac
    
    read -p "Press Enter to continue..."
done