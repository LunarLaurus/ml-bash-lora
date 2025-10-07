#!/bin/bash


# robust download of latest tree-sitter linux binary (arch-aware)
download_tree_sitter() {
    local BUILD_DIR="$SCRIPT_DIR/build"
    local TS_BIN="$BUILD_DIR/tree-sitter"
    local RETRIES=3
    local SLEEP=2
    
    mkdir -p "$BUILD_DIR" || {
        echo "[ERROR] Failed to create build dir: $BUILD_DIR"
        return 1
    }
    
    # quick sanity: check free space (at least 10MB)
    local avail_kb
    avail_kb=$(df --output=avail -k "$BUILD_DIR" 2>/dev/null | tail -n1 || echo 0)
    if [ -z "$avail_kb" ]; then avail_kb=0; fi
    if [ "$avail_kb" -lt 10240 ]; then
        echo "[ERROR] Not enough disk space in $BUILD_DIR (need ~10MB). Available: ${avail_kb}KB"
        return 1
    fi
    
    # find asset URL for local arch (x64/arm64)
    local arch asset_sub TS_URL tmpfile rc
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64) asset_sub="tree-sitter-linux-x64" ;;
        aarch64|arm64) asset_sub="tree-sitter-linux-arm64" ;;
        armv7l) asset_sub="tree-sitter-linux-armv7" ;;
        *)
            echo "[ERROR] Unsupported architecture: $arch"
            return 1
        ;;
    esac
    
    TS_URL=$(curl -s "https://api.github.com/repos/tree-sitter/tree-sitter/releases/latest" \
    | grep "browser_download_url" | grep "$asset_sub" | head -n1 | cut -d '"' -f 4 || true)
    
    if [ -z "$TS_URL" ]; then
        echo "[ERROR] Could not find tree-sitter release asset for '$asset_sub'"
        return 1
    fi
    
    tmpfile="${TS_BIN}.part"
    
    # try download with retries
    for i in $(seq 1 $RETRIES); do
        echo "[INFO] Downloading tree-sitter (attempt $i/$RETRIES) from: $TS_URL"
        # use -f (fail on HTTP error), -L follow redirects, -C - to resume if partial downloaded
        curl -fL --retry 3 --retry-delay 2 -o "$tmpfile" "$TS_URL"
        rc=$?
        if [ $rc -eq 0 ]; then
            # atomic move
            mv -f "$tmpfile" "$TS_BIN"
            chmod +x "$TS_BIN"
            echo "[INFO] Download successful: $TS_BIN"
            return 0
        else
            echo "[WARN] Download failed (curl exit $rc). Retrying in ${SLEEP}s..."
            sleep $SLEEP
        fi
    done
    
    echo "[ERROR] Failed to download tree-sitter after $RETRIES attempts."
    [ -f "$tmpfile" ] && rm -f "$tmpfile"
    return 1
}

# -------------------------------
# Extract code dataset from a repo (updated for new tree-sitter API)
# -------------------------------
extract_code_dataset() {
    list_repos
    read -rp "Enter repo index or folder name to extract (or 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    local url folder
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        if (( sel < 0 || sel >= ${#poke_repos_mainline[@]} )); then
            echo -e "${BRED}Index out of range${NC}"
            return 1
        fi
        url="${poke_repos_mainline[$sel]}"
        folder="$(repo_folder_from_url "$url")"
    else
        folder="$sel"
        # try to find matching URL for info
        for u in "${poke_repos_mainline[@]}"; do
            if [ "$(repo_folder_from_url "$u")" = "$sel" ]; then
                url="$u"
                break
            fi
        done
    fi
    
    if [ ! -d "$folder" ]; then
        echo -e "${BRED}Folder '$folder' does not exist locally. Clone it first.${NC}"
        return 1
    fi
    
    ensure_python_cmd || { echo -e "${RED}Python not found.${NC}"; return 1; }
    
    # Check Python dependencies
    local missing_deps=()
    for dep in tree_sitter tqdm; do
        if ! "$PYTHON_CMD" -c "import $dep" &>/dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${BRED}Missing Python dependencies: ${missing_deps[*]}${NC}"
        SUGGESTED="${missing_deps[*]}"
        echo "Attempting to install automatically: ${PIP_CMD[@]} install --upgrade $SUGGESTED"
        "${PIP_CMD[@]}" install --upgrade $SUGGESTED || {
            echo -e "${BRED}Automatic installation failed. Please install manually:${NC} pip install $SUGGESTED"
            return 1
        }
        echo -e "${BGREEN}Dependencies installed successfully.${NC}"
    fi
    
    # -------------------------------
    # Prepare Tree-sitter languages (C and ASM)
    # -------------------------------
    mkdir -p "$SCRIPT_DIR/build"
    download_tree_sitter || {
        echo "[ERROR] Could not obtain tree-sitter CLI. Aborting."
        return 1
    }
    
    export PATH="$SCRIPT_DIR/build:$PATH"
    # verify it runs
    "$SCRIPT_DIR/build/tree-sitter" --version || { echo "[ERROR] tree-sitter binary failed to run"; return 1; }
    
    
    # Ensure grammar repos (clone if missing)
    [ ! -d "$TS_C_REPO" ] && git clone https://github.com/tree-sitter/tree-sitter-c.git "$TS_C_REPO"
    [ ! -d "$TS_ASM_REPO" ] && git clone https://github.com/RubixDev/tree-sitter-asm.git "$TS_ASM_REPO"
    
    # Generate parser code for each grammar (creates src/parser.c inside grammar)
    "$TS_BIN" generate "$TS_C_REPO"
    "$TS_BIN" generate "$TS_ASM_REPO"
    
    # Build shared library using the generated parser.c files
    echo "[INFO] Building shared library at $TS_SO"
    gcc -shared -o "$TS_SO" \
    "$TS_C_REPO/src/parser.c" \
    "$TS_ASM_REPO/src/parser.c" \
    -fPIC 2>/tmp/ts_gcc_build.log || {
        echo "[ERROR] gcc failed to build my-languages.so. See /tmp/ts_gcc_build.log"
        sed -n '1,200p' /tmp/ts_gcc_build.log
        exit 1
    }
    
    mkdir -p "$BUILD_DIR"
    echo "[INFO] Tree-sitter shared library built: $TS_SO"
    
    # Set output file
    local output_file="${folder}_dataset.jsonl"
    
    echo -e "${BGREEN}Extracting code dataset from '$folder' into '$output_file'...${NC}"
    "$PYTHON_CMD" "$SCRIPT_DIR/process-repo.py" "$folder" --out "$output_file" --ts_so "$TS_SO"
    
    if [ $? -eq 0 ]; then
        echo -e "${BGREEN}Extraction complete! Dataset saved to '$output_file'${NC}"
    else
        echo -e "${BRED}Extraction failed${NC}"
    fi
}

# -------------------------------
# Fine-tune LoRA on a repo dataset
# -------------------------------
train_repo_lora() {
    list_repos
    read -rp "Enter repo index or folder name to train LoRA (or 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    local folder dataset output
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        if (( sel < 0 || sel >= ${#poke_repos_mainline[@]} )); then
            echo -e "${BRED}Index out of range${NC}"
            return 1
        fi
        folder="$(repo_folder_from_url "${poke_repos_mainline[$sel]}")"
    else
        folder="$sel"
    fi
    
    dataset="${folder}_dataset.jsonl"
    output="./lora_${folder}"
    
    if [ ! -f "$dataset" ]; then
        echo -e "${BRED}Dataset file '$dataset' not found. Run extraction first.${NC}"
        return 1
    fi
    
    # Check Python dependencies
    local missing_deps=()
    for dep in transformers datasets peft torch tqdm; do
        if ! "$PYTHON_CMD" -c "import $dep" &>/dev/null; then
            missing_deps+=("$dep")
        fi
    done
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${BRED}Missing Python dependencies: ${missing_deps[*]}${NC}"
        SUGGESTED="${missing_deps[*]}"
        echo "Attempting to install automatically: ${PIP_CMD[@]} install --upgrade $SUGGESTED"
        "${PIP_CMD[@]}" install --upgrade $SUGGESTED || {
            echo -e "${BRED}Automatic installation failed. Please install manually:${NC} pip install $SUGGESTED"
            return 1
        }
        echo -e "${BGREEN}Dependencies installed successfully.${NC}"
    fi
    
    echo -e "${BGREEN}Starting LoRA fine-tuning for '$folder'...${NC}"
    "$PYTHON_CMD" "$SCRIPT_DIR/lora_train_repo.py" "$dataset" --output_dir "$output"
    echo -e "${BGREEN}LoRA training finished. Adapter saved to '$output'${NC}"
}