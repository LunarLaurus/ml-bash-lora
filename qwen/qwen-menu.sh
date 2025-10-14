#!/bin/bash
set -euo pipefail

# -------------------------
# Source helpers
# -------------------------
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/conda/env_manager.sh"
source "$PROJECT_ROOT/git/git_data.sh"
source "$PROJECT_ROOT/git/git_utils.sh"
source "$PROJECT_ROOT/git/python_utils.sh"

# -------------------------
# Globals
# -------------------------
CURRENT_REPO_PATH=""
DATA_DIR="data"

# -------------------------
# Check repo selected
# -------------------------
current_repo_path_check() {
    if [ -z "$CURRENT_REPO_PATH" ]; then
        error "No repository selected. Please select a repository first."
        return 1
    fi
}

# -------------------------
# Ensure data dir exists
# -------------------------
ensure_repo_data_dir() {
    if ! current_repo_path_check; then return 1; fi
    local dir="$CURRENT_REPO_PATH/$DATA_DIR"
    mkdir -p "$dir"
}

# -------------------------
# Cleanup repo-specific JSONL
# -------------------------
cleanup_jsonl() {
    ensure_repo_data_dir
    warn "Removing all .jsonl files in $CURRENT_REPO_PATH/$DATA_DIR ..."
    find "$CURRENT_REPO_PATH/$DATA_DIR" -type f -name "*.jsonl" -exec rm -f {} +
    info "Done."
}

# -------------------------
# Check if a file exists in repo-specific data dir
# -------------------------
check_file() {
    ensure_repo_data_dir
    local file="$1"
    if [ -z "$file" ]; then
        echo "Usage: check_file <filename>"
        return 1
    fi
    
    if [ -f "$CURRENT_REPO_PATH/$DATA_DIR/$file" ]; then
        echo "ok"
        return 0
    else
        echo "missing"
        return 1
    fi
}

# -------------------------
# Install dependencies
# -------------------------
install_dependencies() {
    ensure_repo_data_dir
    ensure_python_cmd || { error "Python not found. Activate env first."; return 1; }
    
    check_python_deps numpy torch torchvision torchaudio transformers datasets peft tqdm numpy scipy sklearn tiktoken google.protobuf bitsandbytes accelerate safetensors
    if [ ${#MISSING_PY_DEPS[@]} -gt 0 ]; then
        warn "Missing Python dependencies: ${MISSING_PY_DEPS[*]}"
        auto_install_python_deps || {
            error "Automatic installation failed. Install manually: pip install ${MISSING_PY_DEPS[*]}"
            return 1
        }
        info "${BGREEN}Dependencies installed successfully.${NC}"
    fi
}

# -------------------------
# Run a Python script with repo context
# Optional: $2 is expected input file in repo data dir
# -------------------------
run_script() {
    local script_path="$1"
    local expected_input="${2:-}"
    shift
    [ -n "$expected_input" ] && shift
    
    current_repo_path_check || return 1
    
    if [ -n "$expected_input" ]; then
        if ! check_file "$expected_input"; then
            error "Required input '$expected_input' missing in ${CURRENT_REPO_PATH}/${DATA_DIR}. Aborting."
            return 1
        fi
    fi
    
    ensure_repo_data_dir
    info "Running Script: $script_path for $CURRENT_REPO_PATH"
    "$PYTHON_CMD" "$script_path" "$CURRENT_REPO_PATH" "$@"
}

# -------------------------
# Select project
# -------------------------
select_project() {
    prompt_repo_selection || { info "Cancelled"; return 1; }
    
    # REPO_SEL is set by prompt_repo_selection
    if ! resolve_selection "$REPO_SEL"; then
        error "Invalid selection"
        return 1
    fi
    
    # resolve_selection sets:
    #   REPO_SEL_URL, REPO_SEL_FOLDER, REPO_LOCAL_PATH
    # Use the helper's local path (REPO_LOCAL_PATH is REPO_BASE_DIR/<folder>)
    CURRENT_REPO_PATH="$REPO_LOCAL_PATH"
    
    # ensure the local path exists
    if [ ! -d "$CURRENT_REPO_PATH" ]; then
        error "Repo folder $CURRENT_REPO_PATH does not exist locally. Clone it first."
        return 1
    fi
    
    info "Selected project: $CURRENT_REPO_PATH"
}


# Prompt selection at script start
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
        *) error "Invalid choice. Please try again." ;;
    esac
    
    read -p "Press Enter to continue..."
done
