#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
update_script_dir 2
source "$PROJECT_ROOT/conda/env_manager.sh"

# ------------------------------
# Apt install helpers (single or grouped)
# ------------------------------
install_apt_single() {
    pkg="$1"
    [ -z "$pkg" ] && return 1
    update_apt_cache
    sudo apt-get install -y "$pkg"
}

install_apt_group() {
    # Accepts a name + list of pkgs
    group_name="$1"; shift
    pkgs=("$@")
    echo -e "${GREEN}Installing ${group_name}: ${pkgs[*]}${NC}"
    update_apt_cache
    sudo apt-get install -y "${pkgs[@]}"
}

# Predefined groups (conservative, small)
install_group_essentials() {
    install_apt_group "Essentials" git curl wget unzip
}

install_group_devtools() {
    install_apt_group "Development Tools" build-essential cmake pkg-config
}

install_group_media() {
    install_apt_group "Media Tools" ffmpeg
}

install_group_utils() {
    install_apt_group "Utilities" htop ncdu tree jq
}

install_git_lfs() {
    install_apt_single git-lfs
    if command -v git-lfs &>/dev/null; then
        git lfs install --skip-repo || true
    fi
}

# ------------------------------
# Python (conda env only) helpers
# ------------------------------
install_python_packages_conda() {
    if ! get_active_env; then
        error_no_env
        return 1
    fi
    # packages passed as arguments
    if [ "$#" -eq 0 ]; then
        echo -e "${YELLOW}No packages specified.${NC}"
        return 1
    fi
    echo -e "${BCYAN}Installing Python packages into conda env '$CURRENT_ENV': $*${NC}"
    # Use conda run pip to avoid activating subshell
    conda run -n "$CURRENT_ENV" python -m pip install --upgrade "$@" || {
        echo -e "${YELLOW}pip install reported errors. Try running inside the env manually for more details.${NC}"
        return 1
    }
    echo -e "${BCYAN}Done.${NC}"
}

# Some small curated pip groups (opt-in)
pip_group_text_processing() {
    install_python_packages_conda "sentencepiece" "tokenizers" "tiktoken"
}

pip_group_monitoring() {
    install_python_packages_conda "psutil" "prometheus-client"
}

pip_group_utils() {
    install_python_packages_conda "humanize" "regex"
}

# ------------------------------
# Find large projects
# ------------------------------
find_large_projects() {
    root="${1:-$HOME}"
    depth="${2:-2}"
    topn="${3:-20}"

    if [ ! -d "$root" ]; then
        echo -e "${RED}Root path '$root' does not exist.${NC}"
        return 1
    fi

    echo -e "${GREEN}Scanning for largest directories under: $root (depth=$depth)${NC}"
    tmpfile=$(mktemp)
    ( cd "$root" && du -B1 -d "$depth" 2>/dev/null ) >"$tmpfile"
    if [ ! -s "$tmpfile" ]; then
        echo -e "${YELLOW}No results from du; permission issues or empty directory.${NC}"
        rm -f "$tmpfile"
        return 1
    fi
    sort -nr "$tmpfile" | head -n "$topn" | awk '{ printf "%12s  %s\n", $1, $2 }' | \
        while read -r bytes path; do
            if command -v numfmt &>/dev/null; then
                h=$(numfmt --to=iec --suffix=B "$bytes")
            else
                h="$bytes"
            fi
            echo -e "${GREEN}$h${NC}  $root/$path"
        done
    rm -f "$tmpfile"
}

# ------------------------------
# List LoRA / RAG packages (conda env only)
# ------------------------------
list_lora_lib_versions() {
    if ! get_active_env; then
        error_no_env
        return 1
    fi
    echo -e "${BCYAN}LoRA-related package versions in '$CURRENT_ENV':${NC}"
    conda run -n "$CURRENT_ENV" python - <<'PY'
import importlib
pkgs = ["transformers","peft","bitsandbytes","accelerate","datasets","safetensors"]
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, "__version__", None) or getattr(m, "version", None) or "unknown"
        print(f"{p}: {v}")
    except Exception as e:
        print(f"{p}: NOT INSTALLED ({e.__class__.__name__})")
PY
}

list_rag_lib_versions() {
    if ! get_active_env; then
        error_no_env
        return 1
    fi
    echo -e "${BCYAN}RAG-related package versions in '$CURRENT_ENV':${NC}"
    conda run -n "$CURRENT_ENV" python - <<'PY'
import importlib
pkgs = ["faiss","faiss_cpu","sentence_transformers","langchain","sentencepiece"]
# normalize import names
for p in pkgs:
    key = p.replace("-", "_")
    try:
        m = importlib.import_module(key)
        v = getattr(m, "__version__", None) or getattr(m, "version", None) or "unknown"
        print(f"{p}: {v}")
    except Exception as e:
        print(f"{p}: NOT INSTALLED ({e.__class__.__name__})")
PY
}

# ------------------------------
# Menu
# ------------------------------
while true; do
    echo -e "\n${CYAN}=== Diagnostics & Utilities (selective) ===${NC}"
    echo "---- Apt single installs ----"
    echo "10) Install git"
    echo "11) Install git-lfs"
    echo "12) Install curl/wget"
    echo "13) Install ffmpeg"
    echo "14) Install build-essential"
    echo "---- Apt small groups ----"
    echo "20) Install Essentials (git, curl, wget, unzip)"
    echo "21) Install Devtools (build-essential, cmake, pkg-config)"
    echo "22) Install Media tools (ffmpeg)"
    echo "23) Install Utilities (htop, ncdu, tree, jq)"
    echo "---- Pip into conda env (conda required) ----"
    echo "30) Install pip packages: text-processing group (sentencepiece, tokenizers, tiktoken)"
    echo "31) Install pip packages: monitoring group (psutil, prometheus-client)"
    echo "32) Install pip utilities (humanize, regex)"
    echo "---- Other ----"
    echo "40) Find largest projects/directories"
    echo "50) List LoRA lib versions (in active conda env)"
    echo "51) List RAG lib versions (in active conda env)"
    echo -e "${BRED}0) Back to Main Menu${NC}"

    read -rp "Choice: " choice
    case $choice in
        10) install_apt_single git ;;
        11) install_git_lfs ;;
        12) install_apt_group "Download tools" curl wget ;;
        13) install_apt_single ffmpeg ;;
        14) install_apt_single build-essential ;;

        20) install_group_essentials ;;
        21) install_group_devtools ;;
        22) install_group_media ;;
        23) install_group_utils ;;

        30) pip_group_text_processing ;;
        31) pip_group_monitoring ;;
        32) pip_group_utils ;;

        40)
            read -rp "Path to scan (default: $HOME): " scanroot
            scanroot=${scanroot:-$HOME}
            read -rp "Depth to scan (default: 2): " depth
            depth=${depth:-2}
            read -rp "How many results to show (default: 20): " topn
            topn=${topn:-20}
            find_large_projects "$scanroot" "$depth" "$topn"
            ;;
        50) list_lora_lib_versions ;;
        51) list_rag_lib_versions ;;

        0) break ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
