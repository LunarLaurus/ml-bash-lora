#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/conda/env_manager.sh"
source "$PROJECT_ROOT/git/git_data.sh"
source "$PROJECT_ROOT/git/git_utils.sh"
source "$PROJECT_ROOT/git/python_utils.sh"

CURRENT_REPO_PATH=""
DATA_DIR="data"

current_repo_path_check() {
    if [ -z "$CURRENT_REPO_PATH" ]; then
        echo -e "${BRED}No repository selected. Please select a repository first.${NC}"
        return 1
    fi
}

# -------------------------
# Remove all .jsonl files in repo-specific data dir
# -------------------------
cleanup_jsonl() {
    if ! current_repo_path_check; then return 1; fi
    local repo_data_dir="${CURRENT_REPO_PATH}/${DATA_DIR}"
    if [ -d "$repo_data_dir" ]; then
        echo "Removing all .jsonl files in $repo_data_dir..."
        find "$repo_data_dir" -type f -name "*.jsonl" -exec rm -f {} +
        echo "Done."
    else
        echo "Directory $repo_data_dir does not exist."
    fi
}

# -------------------------
# Check if a file exists in repo-specific data dir
# -------------------------
check_file() {
    if ! current_repo_path_check; then return 1; fi
    local file="$1"
    local repo_data_dir="${CURRENT_REPO_PATH}/${DATA_DIR}"
    
    if [ -z "$file" ]; then
        echo "Usage: check_file <filename>"
        return 1
    fi
    
    if [ -f "$repo_data_dir/$file" ]; then
        echo "ok"
        return 0
    else
        echo "missing"
        return 1
    fi
}

# -------------------------
# Install dependencies for current repo
# -------------------------
install_dependencies() {
    if ! current_repo_path_check; then return 1; fi
    ensure_python_cmd || { echo -e "${RED}Python not found. Activate env first.${NC}"; return 1; }
    
    check_python_deps numpy torch torchvision torchaudio transformers datasets peft tqdm numpy scipy sklearn tiktoken google.protobuf bitsandbytes accelerate safetensors
    if [ ${#MISSING_PY_DEPS[@]} -gt 0 ]; then
        warn "${BRED}Missing Python dependencies: ${MISSING_PY_DEPS[*]}${NC}"
        auto_install_python_deps || {
            error "${BRED}Automatic installation failed. Install manually: pip install ${MISSING_PY_DEPS[*]}${NC}"
            return 1
        }
        info "${BGREEN}Dependencies installed successfully.${NC}"
    fi
}

# -------------------------
# Run a script with CURRENT_REPO_PATH as first argument
# Optional: check for input file in repo data dir
# -------------------------
run_script() {
    local script_path="$1"
    local expected_input="${2-}"
    
    shift
    if [ -n "$expected_input" ]; then
        shift  # only shift $2 if it exists
    fi
    
    if ! current_repo_path_check; then
        return 1
    fi
    
    if [ -n "$expected_input" ]; then
        if ! check_file "$expected_input"; then
            echo -e "${BRED}Required input '$expected_input' missing in ${CURRENT_REPO_PATH}/${DATA_DIR}. Aborting.${NC}"
            return 1
        fi
    fi
    
    info "Running Script: $script_path for $CURRENT_REPO_PATH"
    "$PYTHON_CMD" "$script_path" "$CURRENT_REPO_PATH" "$@"
}


# -------------------------
# Project selection
# -------------------------
select_project() {
    if ! prompt_repo_selection; then
        info "Cancelled"
        return 1
    fi
    resolve_selection_to_folder "$REPO_SEL" || return 1
    ensure_python_cmd || { error "Python not found. Activate env first."; return 1; }
    
    # numeric index maps to folder
    if [[ "$REPO_SEL" =~ ^[0-9]+$ ]]; then
        CURRENT_REPO_PATH="$(repo_folder_from_url "${poke_repos_mainline[$REPO_SEL]}")"
    fi
    if [ -z "${CURRENT_REPO_PATH:-}" ]; then
        CURRENT_REPO_PATH="$REPO_SEL"
    fi
    
    info "Selected project: $CURRENT_REPO_PATH"
}

# Prompt project selection at script start
select_project || exit 1

# -------------------------
# Main menu loop
# -------------------------
while true; do
    clear
    echo "[qwen2.5-coder:7b] Lora System Script Menu"
    echo "Current project: ${CURRENT_REPO_PATH:-None}"
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
    echo "99) Cleanup .jsonl files"
    echo "0) Exit"
    read -p "Enter your choice: " choice
    
    case $choice in
        1) run_script qwen/01_index_files.py ;;
        2) run_script qwen/02_parse_code.py "parsed_files.jsonl" ;;
        3) run_script qwen/02b_build_dependency_graphs.py "dep_graph_functions.jsonl" ;;
        4) run_script qwen/03_enrich_with_model.py "parsed_functions.jsonl" ;;
        5) run_script qwen/04_link_headers.py "enriched_functions.jsonl" ;;
        6) run_script qwen/05_generate_qna.py "enriched_functions.jsonl" ;;
        7) run_script qwen/06_train_lora.py "qna_train.jsonl" ;;
        
        8) run_script qwen/07_embed_code.py "lora_adapter.jsonl" ;;
        9) run_script qwen/08_query_system.py "embeddings.jsonl" ;;
        10) run_script qwen/09_evaluate.py "qna_dataset.jsonl" ;;
        21) install_dependencies ;;
        99) cleanup_jsonl ;;
        0) break ;;
        *) echo "Invalid choice. Please try again." ;;
    esac
    
    read -p "Press Enter to continue..."
done
