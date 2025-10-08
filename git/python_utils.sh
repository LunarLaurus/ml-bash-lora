#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TS_BIN="${BUILD_DIR}/tree-sitter"
TS_SO="${BUILD_DIR}/my-languages.so"

# Configurable grammar repo locations (overrideable earlier if needed)
: "${TS_C_REPO:=${SCRIPT_DIR}/third_party/tree-sitter-c}"
: "${TS_ASM_REPO:=${SCRIPT_DIR}/third_party/tree-sitter-asm}"

# ----------------------
# Small helper functions
# ----------------------


# check required commands exist (returns nonzero if any are missing)
require_cmds() {
    local miss=0 cmd
    for cmd in "$@"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            error "Required command not found: $cmd"
            miss=1
        fi
    done
    return $miss
}

# ensure build dir exists and has space (~10MB)
ensure_build_dir() {
    mkdir -p "$BUILD_DIR" || { error "Could not create $BUILD_DIR"; return 1; }
    local avail_kb
    avail_kb=$(df --output=avail -k "$BUILD_DIR" 2>/dev/null | tail -n1 || echo 0)
    if [ -z "$avail_kb" ]; then avail_kb=0; fi
    if [ "$avail_kb" -lt 10240 ]; then
        error "Insufficient disk space in $BUILD_DIR (~10MB required). Available: ${avail_kb}KB"
        return 1
    fi
    return 0
}

# Map uname -m -> expected asset substring
arch_asset_sub() {
    case "$(uname -m)" in
        x86_64|amd64) printf '%s' "tree-sitter-linux-x64" ;;
        aarch64|arm64) printf '%s' "tree-sitter-linux-arm64" ;;
        armv7l) printf '%s' "tree-sitter-linux-armv7" ;;
        *) return 1 ;;
    esac
}

# Fetch release JSON, using GITHUB_TOKEN if provided. Output to file path arg1.
fetch_release_json() {
    local out="$1"
    local api_url="https://api.github.com/repos/tree-sitter/tree-sitter/releases/latest"
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        curl -fsSL -H "Authorization: Bearer ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json" "$api_url" -o "$out"
    else
        curl -fsSL -H "Accept: application/vnd.github+json" "$api_url" -o "$out"
    fi
}

# Pick best asset URL from release JSON; fallback to simple grep scraping if jq missing.
choose_asset_url() {
    local jsonfile="$1"
    local want="$2"
    if command -v jq >/dev/null 2>&1; then
        jq -r --arg want "$want" '
          .assets[]?.browser_download_url
          | select(test($want) or (test("linux") and (test("x86|x64|arm"))))
        ' "$jsonfile" 2>/dev/null | head -n1 || true
    else
        # crude grep fallback: find browser_download_url lines and match substring
        grep '"browser_download_url"' "$jsonfile" | cut -d'"' -f4 | grep "$want" | head -n1 || \
        grep '"browser_download_url"' "$jsonfile" | cut -d'"' -f4 | grep -i linux | head -n1 || true
    fi
}

# Download with resume into BUILD_DIR, atomic move to TS_BIN
download_with_resume() {
    local url="$1"
    local tmpfile
    tmpfile="$(mktemp "${BUILD_DIR}/tree-sitter.part.XXXXX")" || return 1
    trap 'rm -f "$tmpfile"' EXIT
    
    local tries=0 max=5
    while [ $tries -lt $max ]; do
        tries=$((tries + 1))
        info "Downloading tree-sitter (attempt $tries/$max) from: $url"
        # -f fail on HTTP errors, -L follow redirects, -C - resume
        if curl -fL --retry 5 --retry-delay 2 -C - -o "$tmpfile" "$url"; then
            mv -f "$tmpfile" "$TS_BIN"
            chmod +x "$TS_BIN"
            trap - EXIT
            info "Downloaded tree-sitter -> $TS_BIN"
            return 0
        else
            warn "Download failed (curl exit). Retrying..."
            sleep 2
        fi
    done
    
    rm -f "$tmpfile"
    trap - EXIT
    return 1
}

# Public: install tree-sitter CLI (small composed steps)
install_tree_sitter() {
    require_cmds curl mktemp mv chmod uname || return 1
    ensure_build_dir || return 1
    local asset_sub jsonfile asset_url
    asset_sub="$(arch_asset_sub)" || { error "Unsupported architecture $(uname -m)"; return 1; }
    
    jsonfile="$(mktemp)" || return 1
    if fetch_release_json "$jsonfile"; then
        asset_url="$(choose_asset_url "$jsonfile" "$asset_sub")"
        rm -f "$jsonfile"
    else
        rm -f "$jsonfile"
        # fallback: try releases/latest HTML scraping (less reliable)
        asset_url=$(curl -fsSL "https://github.com/tree-sitter/tree-sitter/releases/latest" \
            | grep -Eo 'https://github.com/tree-sitter/tree-sitter/releases/download[^"]+' \
        | grep "$asset_sub" | head -n1 || true)
    fi
    
    if [ -z "$asset_url" ]; then
        error "Could not find asset URL for $asset_sub"
        return 1
    fi
    
    download_with_resume "$asset_url" || { error "Failed to download tree-sitter"; return 1; }
    
    # quick run check
    if ! "$TS_BIN" --version >/dev/null 2>&1; then
        error "Downloaded tree-sitter binary couldn't run."
        return 1
    fi
    
    return 0
}

# Ensure grammar repos exist (clone if missing)
ensure_grammar_repo() {
    local repo_url="$1" dest="$2"
    if [ -d "$dest" ]; then
        info "Grammar repo present: $dest"
        return 0
    fi
    require_cmds git || return 1
    info "Cloning grammar repo $repo_url -> $dest"
    git clone "$repo_url" "$dest" || { error "Failed to clone $repo_url"; return 1; }
}

# Generate parser.c with the installed tree-sitter binary (run inside repo dir)
generate_parser() {
    local repo_dir="$1"
    if [ ! -x "$TS_BIN" ]; then
        error "tree-sitter binary not found at $TS_BIN"
        return 1
    fi
    if [ ! -d "$repo_dir" ]; then
        error "Grammar repo dir not found: $repo_dir"
        return 1
    fi
    # run generate inside the grammar dir (safer)
    ( set -x; cd "$repo_dir" && "$TS_BIN" generate ) || { error "tree-sitter generate failed for $repo_dir"; return 1; }
}

# Build the combined shared library from parser.c files
build_shared_library() {
    local c_repo="$1" asm_repo="$2" out_so="$3"
    require_cmds gcc || { error "gcc required to build shared library"; return 1; }
    local c_parser="${c_repo}/src/parser.c"
    local asm_parser="${asm_repo}/src/parser.c"
    if [ ! -f "$c_parser" ] || [ ! -f "$asm_parser" ]; then
        error "Missing parser.c in one of grammar repos ($c_parser or $asm_parser)"
        return 1
    fi
    
    info "Building shared library: $out_so"
    local log="/tmp/ts_gcc_build.log"
    rm -f "$log"
    if gcc -shared -o "$out_so" "$c_parser" "$asm_parser" -fPIC 2>"$log"; then
        info "Built shared library: $out_so"
        return 0
    else
        error "gcc build failed. See $log (showing head)"
        sed -n '1,200p' "$log" >&2
        return 1
    fi
}

# -------------------------------
# Smaller helpers for repo selection, python deps, running scripts
# -------------------------------

prompt_repo_selection() {
    # expects list_repos to exist and populate poke_repos_mainline
    list_repos
    read -r -p "Enter repo index or folder name (or 'q' to cancel): " REPO_SEL
    [ "${REPO_SEL:-}" = "q" ] && return 1
    return 0
}

resolve_selection_to_folder() {
    local sel="$1"
    FOLDER_URL=""
    FOLDER_PATH=""
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        if (( sel < 0 || sel >= ${#poke_repos_mainline[@]} )); then
            error "Index out of range"
            return 1
        fi
        FOLDER_URL="${poke_repos_mainline[$sel]}"
        FOLDER_PATH="$(repo_folder_from_url "$FOLDER_URL")"
    else
        FOLDER_PATH="$sel"
        for u in "${poke_repos_mainline[@]}"; do
            if [ "$(repo_folder_from_url "$u")" = "$sel" ]; then
                FOLDER_URL="$u"
                break
            fi
        done
    fi
    return 0
}

ensure_folder_exists() {
    if [ -z "${FOLDER_PATH:-}" ]; then
        error "Internal: FOLDER_PATH not set"
        return 1
    fi
    if [ ! -d "$FOLDER_PATH" ]; then
        error "Folder '$FOLDER_PATH' does not exist locally. Clone it first."
        return 1
    fi
    return 0
}

check_python_deps() {
    MISSING_PY_DEPS=()
    local dep
    for dep in "$@"; do
        if ! "$PYTHON_CMD" -c "import $dep" &>/dev/null; then
            MISSING_PY_DEPS+=("$dep")
        fi
    done
}

auto_install_python_deps() {
    if [ ${#MISSING_PY_DEPS[@]:-0} -eq 0 ]; then
        return 0
    fi
    if [ -z "${PIP_CMD[*]:-}" ]; then
        warn "No PIP_CMD configured; cannot auto-install python deps"
        return 1
    fi
    info "Missing Python dependencies: ${MISSING_PY_DEPS[*]}"
    info "Attempting to install automatically: ${PIP_CMD[*]} install --upgrade ${MISSING_PY_DEPS[*]}"
    "${PIP_CMD[@]}" install --upgrade "${MISSING_PY_DEPS[@]}" || {
        warn "Automatic installation failed"
        return 1
    }
    return 0
}

prepare_tree_sitter_environment() {
    ensure_build_dir || return 1
    
    if [ ! -x "$TS_BIN" ]; then
        info "tree-sitter CLI not found; installing..."
        install_tree_sitter || { error "Could not install tree-sitter"; return 1; }
    fi
    
    if ! "$TS_BIN" --version >/dev/null 2>&1; then
        error "tree-sitter binary failed to run"
        return 1
    fi
    
    ensure_grammar_repo "https://github.com/tree-sitter/tree-sitter-c.git" "$TS_C_REPO" || return 1
    ensure_grammar_repo "https://github.com/RubixDev/tree-sitter-asm.git" "$TS_ASM_REPO" || return 1
    
    generate_parser "$TS_C_REPO" || return 1
    generate_parser "$TS_ASM_REPO" || return 1
    
    build_shared_library "$TS_C_REPO" "$TS_ASM_REPO" "$TS_SO" || return 1
    
    return 0
}

run_python_script() {
    local script_path="$1"; shift
    if [ ! -f "$script_path" ]; then
        error "Python script not found: $script_path"
        return 1
    fi
    "$PYTHON_CMD" "$script_path" "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        error "Python script '$script_path' failed (exit $rc)"
    fi
    return $rc
}

extract_code_dataset() {
    if ! prompt_repo_selection; then
        info "Cancelled"
        return 0
    fi
    resolve_selection_to_folder "$REPO_SEL" || return 1
    
    # ensure python cmd exists (fallbacks handled by default)
    if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
        error "Python not found: $PYTHON_CMD"
        return 1
    fi
    
    check_python_deps tree_sitter tqdm
    if [ ${#MISSING_PY_DEPS[@]:-0} -gt 0 ]; then
        echo -e "${BRED}Missing Python dependencies: ${MISSING_PY_DEPS[*]}${NC}"
        auto_install_python_deps || {
            echo -e "${BRED}Automatic installation failed. Please install manually:${NC} pip install ${MISSING_PY_DEPS[*]}"
            return 1
        }
        echo -e "${BGREEN}Dependencies installed successfully.${NC}"
    fi
    
    ensure_folder_exists || return 1
    prepare_tree_sitter_environment || return 1
    
    local output_file="${FOLDER_PATH}_dataset.jsonl"
    info "Extracting code dataset from '$FOLDER_PATH' into '$output_file'..."
    run_python_script "$SCRIPT_DIR/process-repo.py" "$FOLDER_PATH" --out "$output_file" --ts_so "$TS_SO" || {
        error "Extraction failed"
        return 1
    }
    echo -e "${BGREEN}Extraction complete! Dataset saved to '$output_file'${NC}"
    return 0
}

train_repo_lora() {
    if ! prompt_repo_selection; then
        info "Cancelled"
        return 0
    fi
    resolve_selection_to_folder "$REPO_SEL" || return 1
    
    # compute FOLDER_PATH if index selected
    if [[ "$REPO_SEL" =~ ^[0-9]+$ ]]; then
        FOLDER_PATH="$(repo_folder_from_url "${poke_repos_mainline[$REPO_SEL]}")"
    fi
    if [ -z "${FOLDER_PATH:-}" ]; then
        FOLDER_PATH="$REPO_SEL"
    fi
    
    local dataset="${FOLDER_PATH}_dataset.jsonl"
    local output="./lora_${FOLDER_PATH}"
    
    if [ ! -f "$dataset" ]; then
        error "Dataset file '$dataset' not found. Run extraction first."
        return 1
    fi
    
    if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
        error "Python not found: $PYTHON_CMD"
        return 1
    fi
    
    check_python_deps transformers datasets peft torch tqdm
    if [ ${#MISSING_PY_DEPS[@]:-0} -gt 0 ]; then
        echo -e "${BRED}Missing Python dependencies: ${MISSING_PY_DEPS[*]}${NC}"
        auto_install_python_deps || {
            echo -e "${BRED}Automatic installation failed. Please install manually:${NC} pip install ${MISSING_PY_DEPS[*]}"
            return 1
        }
        echo -e "${BGREEN}Dependencies installed successfully.${NC}"
    fi
    
    info "Starting LoRA fine-tuning for '$FOLDER_PATH'..."
    run_python_script "$SCRIPT_DIR/lora_train_repo.py" "$dataset" --output_dir "$output" || {
        error "LoRA training failed"
        return 1
    }
    echo -e "${BGREEN}LoRA training finished. Adapter saved to '$output'${NC}"
    return 0
}

# Optional convenience: preflight check function
preflight_checks() {
    require_cmds curl git gcc "$PYTHON_CMD" || return 1
    ensure_build_dir || return 1
    info "Preflight checks passed"
    return 0
}