#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/conda/env_manager.sh"
update_script_dir 2
source "$SCRIPT_DIR/git_data.sh"

# Helper: get local folder name from .git URL
repo_folder_from_url() {
    local url="$1"
    # strip trailing .git if present, then take basename
    local name="${url##*/}"
    name="${name%.git}"
    echo "$name"
}

# List available repos and whether they exist locally
list_repos() {
    echo -e "\n${BGREEN}Available repos:${NC}"
    local i=0
    for url in "${poke_repos_mainline[@]}"; do
        local folder
        folder="$(repo_folder_from_url "$url")"
        local exists="no"
        if [ -d "$folder/.git" ]; then exists="yes"; fi
        printf " %2d) %s -> %s (local: %s)\n" "$i" "$folder" "$url" "$exists"
        i=$((i + 1))
    done
}

# Clone a single repo (by index or name)
clone_repo() {
    list_repos
    read -rp "Enter repo index or folder name to clone (or 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    local url=""
    # if numeric index
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        if (( sel < 0 || sel >= ${#poke_repos_mainline[@]} )); then
            echo -e "${BRED}Index out of range${NC}"
            return 1
        fi
        url="${poke_repos_mainline[$sel]}"
    else
        # try match by folder name
        for u in "${poke_repos_mainline[@]}"; do
            if [ "$(repo_folder_from_url "$u")" = "$sel" ]; then
                url="$u"
                break
            fi
        done
        if [ -z "$url" ]; then
            echo -e "${BRED}No repo found matching '$sel'${NC}"
            return 1
        fi
    fi
    
    local folder
    folder="$(repo_folder_from_url "$url")"
    if [ -d "$folder" ]; then
        echo -e "${BRED}Folder '$folder' already exists. Skipping clone.${NC}"
        return 1
    fi
    
    echo -e "${BGREEN}Cloning $url into ./$folder ...${NC}"
    git clone "$url" "$folder"
    echo -e "${BGREEN}Clone finished.${NC}"
}

# Clone all repos in the array (skip existing)
clone_all_repos() {
    for url in "${poke_repos_mainline[@]}"; do
        folder="$(repo_folder_from_url "$url")"
        if [ -d "$folder/.git" ]; then
            echo -e "${BRED}Skipping $folder (already exists)${NC}"
            continue
        fi
        echo -e "${BGREEN}Cloning $folder ...${NC}"
        git clone "$url" "$folder" || {
            echo -e "${BRED}Failed to clone $url${NC}"
        }
    done
    echo -e "${BGREEN}Done cloning all repos.${NC}"
}

# Run git pull in a repo
update_repo() {
    list_repos
    read -rp "Enter repo index or folder name to update (or 'all' to update all, 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    if [ "$sel" = "all" ]; then
        for url in "${poke_repos_mainline[@]}"; do
            folder="$(repo_folder_from_url "$url")"
            if [ -d "$folder/.git" ]; then
                echo -e "${BGREEN}Updating $folder ...${NC}"
                (cd "$folder" && git pull --ff-only) || echo -e "${BRED}Failed to update $folder${NC}"
            else
                echo -e "${BRED}Skipping $folder (not cloned)${NC}"
            fi
        done
        return 0
    fi
    
    local url=""
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        if (( sel < 0 || sel >= ${#poke_repos_mainline[@]} )); then
            echo -e "${BRED}Index out of range${NC}"
            return 1
        fi
        url="${poke_repos_mainline[$sel]}"
    else
        for u in "${poke_repos_mainline[@]}"; do
            if [ "$(repo_folder_from_url "$u")" = "$sel" ]; then
                url="$u"
                break
            fi
        done
    fi
    if [ -z "$url" ]; then
        echo -e "${BRED}Repo not found${NC}"
        return 1
    fi
    folder="$(repo_folder_from_url "$url")"
    if [ ! -d "$folder/.git" ]; then
        echo -e "${BRED}Repo '$folder' not cloned (folder missing)${NC}"
        return 1
    fi
    
    echo -e "${BGREEN}Running git pull in $folder ...${NC}"
    (cd "$folder" && git pull --ff-only) || echo -e "${BRED}git pull failed for $folder${NC}"
}

# Show git status for a repo
repo_status() {
    list_repos
    read -rp "Enter repo index or folder name to show status (or 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    local url=""
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        if (( sel < 0 || sel >= ${#poke_repos_mainline[@]} )); then
            echo -e "${BRED}Index out of range${NC}"
            return 1
        fi
        url="${poke_repos_mainline[$sel]}"
    else
        for u in "${poke_repos_mainline[@]}"; do
            if [ "$(repo_folder_from_url "$u")" = "$sel" ]; then
                url="$u"
                break
            fi
        done
    fi
    if [ -z "$url" ]; then
        echo -e "${BRED}Repo not found${NC}"
        return 1
    fi
    folder="$(repo_folder_from_url "$url")"
    if [ ! -d "$folder/.git" ]; then
        echo -e "${BRED}Repo '$folder' not cloned (folder missing)${NC}"
        return 1
    fi
    
    (cd "$folder" && echo -e "${BGREEN}== $folder status ==${NC}" && git status -s)
}

# Delete a repo's local folder (with confirmation)
delete_repo() {
    list_repos
    read -rp "Enter repo index or folder name to delete (local folder only) (or 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    local folder=""
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        if (( sel < 0 || sel >= ${#poke_repos_mainline[@]} )); then
            echo -e "${BRED}Index out of range${NC}"
            return 1
        fi
        url="${poke_repos_mainline[$sel]}"
        folder="$(repo_folder_from_url "$url")"
    else
        # assume folder name
        folder="$sel"
    fi
    
    if [ ! -d "$folder" ]; then
        echo -e "${BRED}Folder '$folder' does not exist locally.${NC}"
        return 1
    fi
    
    read -rp "Are you sure you want to DELETE the local folder '$folder'? This cannot be undone. (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -rf -- "$folder"
        echo -e "${BGREEN}Deleted $folder${NC}"
    else
        echo -e "${BRED}Aborted${NC}"
    fi
}

# Open a subshell in the repo folder
open_repo_shell() {
    list_repos
    read -rp "Enter repo index or folder name to open a subshell in (or 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    local url=""
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        if (( sel < 0 || sel >= ${#poke_repos_mainline[@]} )); then
            echo -e "${BRED}Index out of range${NC}"
            return 1
        fi
        url="${poke_repos_mainline[$sel]}"
    else
        # try to match
        for u in "${poke_repos_mainline[@]}"; do
            if [ "$(repo_folder_from_url "$u")" = "$sel" ]; then
                url="$u"
                break
            fi
        done
        # if still empty, treat sel as folder name
        if [ -z "$url" ]; then
            folder="$sel"
        fi
    fi
    
    if [ -z "${folder:-}" ]; then
        folder="$(repo_folder_from_url "$url")"
    fi
    
    if [ ! -d "$folder" ]; then
        echo -e "${BRED}Folder '$folder' does not exist.${NC}"
        return 1
    fi
    
    echo -e "${BGREEN}Spawning subshell in ./$folder (type exit to return)...${NC}"
    cd "$folder" || return 1
    bash --login
    # return to script directory afterwards
    cd "$SCRIPT_DIR" || true
}

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
    echo "8) Extract code dataset from repo"
    echo "9) Fine-tune LoRA on a repo dataset"
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
        8) extract_code_dataset ;;
        9) train_repo_lora ;;
        0) break ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
