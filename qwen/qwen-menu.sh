#!/bin/bash

# -------------------------
# Source helpers
# -------------------------
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/conda/env_manager.sh"
source "$PROJECT_ROOT/git/git_data.sh"
source "$PROJECT_ROOT/git/git_utils.sh"
source "$PROJECT_ROOT/git/python_utils.sh"

# -------------------------
# Globals & safe defaults for colors
# -------------------------
CURRENT_REPO_PATH=""
DATA_DIR="data"

# -------------------------
# Helpers (reduced duplication)
# -------------------------
current_repo_path_check() {
    if [ -z "${CURRENT_REPO_PATH:-}" ]; then
        error "No repository selected. Please select a repository first."
        return 1
    fi
    return 0
}

ensure_repo_data_dir() {
    current_repo_path_check || return 1
    local dir="$CURRENT_REPO_PATH/$DATA_DIR"
    mkdir -p "$dir"
}

# silent boolean: does file exist under current repo data dir?
file_exists() {
    local file="${1:-}"
    [ -n "${CURRENT_REPO_PATH:-}" ] || return 1
    [ -n "$file" ] || return 1
    [ -f "$CURRENT_REPO_PATH/$DATA_DIR/$file" ]
}

# pretty status: returns colored "ok" / "missing"
file_status_label() {
    local file="$1"
    if file_exists "$file"; then
        printf "%sok%s" "$BGREEN" "$NC"
    else
        printf "%smissing%s" "$BRED" "$NC"
    fi
}

# -------------------------
# Repo selection (uses git helper resolve_selection)
# - if repo missing and URL known, offer to clone it into REPO_BASE_DIR/<folder>
# -------------------------
select_project() {
    prompt_repo_selection || { info "Cancelled"; return 1; }
    
    if ! resolve_selection "$REPO_SEL"; then
        error "Invalid selection"
        return 1
    fi
    
    # resolve_selection sets:
    #   REPO_SEL_URL, REPO_SEL_FOLDER, REPO_LOCAL_PATH
    # Use the helper's local path (REPO_LOCAL_PATH is REPO_BASE_DIR/<folder>)
    if [ -z "${REPO_LOCAL_PATH:-}" ]; then
        error "Internal: REPO_LOCAL_PATH not set by resolve_selection"
        return 1
    fi
    
    # If repo folder missing, offer to clone (only if URL known)
    if [ ! -d "$REPO_LOCAL_PATH" ]; then
        if [ -n "${REPO_SEL_URL:-}" ]; then
            printf "%sRepo '%s' not found locally.%s\n" "$BYELLOW" "$REPO_SEL_FOLDER" "$NC"
            read -rp "Clone ${REPO_SEL_URL} into ${REPO_LOCAL_PATH}? (Y/n) " ans
            ans="${ans:-Y}"
            if [[ "$ans" =~ ^([Yy]|)$ ]]; then
                mkdir -p "$REPO_BASE_DIR"
                info "Cloning ${REPO_SEL_URL} -> ${REPO_LOCAL_PATH} ..."
                if git clone "${REPO_SEL_URL}" "${REPO_LOCAL_PATH}"; then
                    info "Clone completed."
                else
                    error "Clone failed. Aborting selection."
                    return 1
                fi
            else
                error "Repo not available locally. Selection aborted."
                return 1
            fi
        else
            error "No known URL for selection; cannot auto-clone. Clone manually and try again."
            return 1
        fi
    fi
    
    # set CURRENT_REPO_PATH to the absolute repo-local path
    CURRENT_REPO_PATH="$(realpath "$REPO_LOCAL_PATH")"
    info "Selected project: $CURRENT_REPO_PATH"
    return 0
}

# -------------------------
# Cleanup repo-specific JSONL
# -------------------------
cleanup_jsonl() {
    ensure_repo_data_dir || return 1
    warn "Removing all .jsonl files in $CURRENT_REPO_PATH/$DATA_DIR ..."
    find "$CURRENT_REPO_PATH/$DATA_DIR" -type f -name "*.jsonl" -exec rm -f {} +
    info "Done."
}

# -------------------------
# Install dependencies
# -------------------------
install_dependencies() {
    ensure_repo_data_dir || return 1
    ensure_python_cmd || { error "Python not found. Activate env first."; return 1; }
    
    check_python_deps numpy torch torchvision torchaudio transformers datasets peft tqdm numpy scipy sklearn tiktoken google.protobuf bitsandbytes accelerate safetensors
    if [ ${#MISSING_PY_DEPS[@]:-0} -gt 0 ]; then
        warn "Missing Python dependencies: ${MISSING_PY_DEPS[*]}"
        auto_install_python_deps || {
            error "Automatic installation failed. Install manually: pip install ${MISSING_PY_DEPS[*]}"
            return 1
        }
        info "Dependencies installed successfully."
    fi
}

# -------------------------
# Run a Python script with repo context
# Optional: $2 is expected input file in repo data dir (checked before run)
# -------------------------
run_script() {
    local script_path="$1"
    local expected_input="${2:-}"
    shift
    # shift extra only if expected_input supplied (preserve further args)
    if [ -n "$expected_input" ]; then shift || true; fi
    
    current_repo_path_check || return 1
    ensure_repo_data_dir || return 1
    local adjusted_path="$PROJECT_ROOT/$script_path"
    
    # Verify script file exists (avoid failing with cryptic message)
    if [ ! -f "$adjusted_path" ]; then
        error "Script not found: $adjusted_path | ($script_path)"
        return 1
    fi
    if [ -n "$expected_input" ]; then
        if ! file_exists "$expected_input"; then
            error "Required input '$expected_input' missing in ${CURRENT_REPO_PATH}/${DATA_DIR}. Aborting."
            return 1
        fi
    fi
    
    info "Running Script: $adjusted_path for $CURRENT_REPO_PATH"
    run_python_file "$adjusted_path" "$CURRENT_REPO_PATH" || {
        error "Script execution failed"
        return 1
    }
}

# -------------------------
# Menu printing helper (reduces duplication)
# -------------------------
menu_entry() {
    local idx="$1"; shift
    local label="$1"; shift
    local script_path="${1:-}"; shift || true
    local expected_input="${1:-}"
    
    if [ -n "$expected_input" ]; then
        # show colored status
        printf "%2s) %s [%s]\n" "$idx" "$label" "$(file_status_label "$expected_input")"
    else
        printf "%2s) %s\n" "$idx" "$label"
    fi
}

# -------------------------
# Bootstrap: select project at start
# -------------------------
select_project || exit 1

# -------------------------
# Main menu loop
# -------------------------
while true; do
    clear
    echo "[qwen2.5-coder:7b] Lora System Script Menu"
    echo "Current project: ${CURRENT_REPO_PATH:-None}"
    echo "-------------------------"
    
    menu_entry 1  "Index Project Files"
    menu_entry 2  "Parse Indexed Code"                "qwen/02_parse_code.py"               "parsed_files.jsonl"
    menu_entry 3  "Build Dependency Graph"            "qwen/02b_build_dependency_graphs.py" "dep_graph_functions.jsonl"
    menu_entry 4  "Enrich"                            "qwen/03_enrich_with_model.py"        "parsed_functions.jsonl"
    menu_entry 5  "Link Headers"                      "qwen/04_link_headers.py"             "enriched_functions.jsonl"
    menu_entry 6  "Generate QNA"                      "qwen/05_generate_qna.py"             "enriched_functions.jsonl"
    menu_entry 7  "Train LoRA"                        "qwen/06_train_lora.py"               "qna_train.jsonl"
    menu_entry 8  "Embed Code"                        "qwen/07_embed_code.py"               "lora_adapter.jsonl"
    menu_entry 9  "Query System"                      "qwen/08_query_system.py"             "embeddings.jsonl"
    menu_entry 10 "Evaluate"                          "qwen/09_evaluate.py"                 "qna_dataset.jsonl"
    
    echo "21) Install Dependencies"
    echo "99) Cleanup .jsonl files"
    echo " 0) Exit"
    echo "-------------------------"
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
