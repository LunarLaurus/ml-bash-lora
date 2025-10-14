#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"

# Map each candidate path -> parsed version string (populates CUDA_MAP as "ver|path" lines)
_build_version_map() {
    CUDA_MAP=()
    for dir in "${CUDA_CANDIDATES[@]}"; do
        base="$(basename "$dir")"
        # attempt to parse version from directory name
        ver="$(printf '%s' "$base" | sed -E 's/^cuda[-_]?//I; s/^cudatoolkit[-_]?//I; s/[^0-9.].*//')"
        # fallback: use nvcc from that dir if available
        if [ -z "$ver" ] && [ -x "$dir/bin/nvcc" ]; then
            ver="$("$dir/bin/nvcc" --version 2>/dev/null | grep -oE 'release [0-9]+(\.[0-9]+)*' | sed 's/release //; s/,//')"
        fi
        # if still empty, use full path as a final fallback to ensure sortable value (will appear last)
        [ -z "$ver" ] && ver="$dir"
        CUDA_MAP+=("$ver|$dir")
    done
}

detect_cuda() {
    # If we've already got a persisted/selected CUDA and nvcc exists there, don't prompt again.
    if check_existing_cuda; then
        return 0
    fi

    # No persistent CUDA detected — run interactive selection (as before)
    detect_cuda_select || return 1
    update_torch_index_url

    # show nvcc info if available
    if command -v nvcc &>/dev/null; then
        echo
        show_nvcc_version
    else
        error "${RED}nvcc not found in PATH after selection.${NC}"
    fi
}

# Check if CUDA already persisted and valid
check_existing_cuda() {
    if [ -L /usr/local/cuda ]; then
        local cur_target
        cur_target="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
        if [ -n "$cur_target" ] && [ -x "$cur_target/bin/nvcc" ]; then
            info "${GREEN}Using existing persisted CUDA at $cur_target${NC}"
            export CUDA_PATH="$cur_target"
            export CUDA_VER="$("$cur_target/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
            export PATH="$CUDA_PATH/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
            return 0
        fi
    fi
    return 1
}

# Return list of candidate CUDA directories (global array: CUDA_CANDIDATES)
detect_cuda_list() {
    CUDA_CANDIDATES=()
    # Globs to check; add other locations here if you have nonstandard installs
    for g in /usr/local/cuda-* /opt/cuda-* /usr/local/cuda* /opt/cuda*; do
        for p in $g; do
            [ -d "$p" ] || continue
            CUDA_CANDIDATES+=("$(readlink -f "$p")")
        done
    done

    # If /usr/local/cuda exists, include its real target (avoid duplicate)
    if [ -e /usr/local/cuda ]; then
        t=$(readlink -f /usr/local/cuda 2>/dev/null || true)
        [ -n "$t" ] && [ -d "$t" ] && CUDA_CANDIDATES+=("$t")
    fi

    # dedupe preserving order
    if [ "${#CUDA_CANDIDATES[@]}" -gt 0 ]; then
        local uniq=()
        for p in "${CUDA_CANDIDATES[@]}"; do
            [[ " ${uniq[*]} " == *" $p "* ]] || uniq+=("$p")
        done
        CUDA_CANDIDATES=("${uniq[@]}")
    fi
}

# Show found CUDA installations, returns count
list_cuda_installations() {
    detect_cuda_list
    if [ "${#CUDA_CANDIDATES[@]}" -eq 0 ]; then
        error "${RED}No CUDA installations found.${NC}"
        return 0
    fi

    _build_version_map

    info "${GREEN}Found CUDA installations:${NC}"
    # sort by version (semantic sort -V) and print
    IFS=$'\n' sorted=($(printf '%s\n' "${CUDA_MAP[@]}" | sort -t'|' -k1,1 -V))
    local i=1
    for entry in "${sorted[@]}"; do
        ver="${entry%%|*}"
        path="${entry#*|}"
        printf "  %2d) %s -> %s\n" "$i" "$ver" "$path"
        ((i++))
    done

    # expose sorted list externally if needed
    CUDA_MAP_SORTED=("${sorted[@]}")
    return "${#sorted[@]}"
}

list_cuda_candidates() {
    detect_cuda_list
    _build_version_map

    if [ "${#CUDA_MAP[@]}" -eq 0 ]; then
        warn "${YELLOW}No CUDA installations found on disk.${NC}"
        return 1
    fi

    info "${GREEN}Detected CUDA installations:${NC}"
    IFS=$'\n' sorted=($(printf '%s\n' "${CUDA_MAP[@]}" | sort -t'|' -k1,1 -V))
    CUDA_MAP_SORTED=("${sorted[@]}")
    local i=1
    for entry in "${sorted[@]}"; do
        ver="${entry%%|*}"
        path="${entry#*|}"
        printf "  %2d) %s -> %s\n" "$i" "$ver" "$path"
        ((i++))
    done
}

select_and_persist_cuda() {
    list_cuda_candidates || return 1

    local default_index="${#CUDA_MAP_SORTED[@]}"  # highest version
    read -rp "Select CUDA version [default: highest]: " choice

    if [ -z "$choice" ]; then
        choice_index=$default_index
    else
        if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#CUDA_MAP_SORTED[@]}" ]; then
            warn "${YELLOW}Invalid choice, using default.${NC}"
            choice_index=$default_index
        else
            choice_index=$choice
        fi
    fi

    selected="${CUDA_MAP_SORTED[$((choice_index-1))]}"
    CUDA_VER="${selected%%|*}"
    CUDA_PATH="${selected#*|}"

    echo "Linking /usr/local/cuda -> $CUDA_PATH"
    sudo ln -sfn "$CUDA_PATH" /usr/local/cuda

    export CUDA_PATH
    export CUDA_VER
    export PATH="/usr/local/cuda/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    update_torch_index_url

    set_cuda_env_persistent "$CUDA_PATH"
	# Apply immediately to current shell
	if [ -f /etc/profile.d/cuda.sh ]; then
		source /etc/profile.d/cuda.sh
	fi

    info "${GREEN}CUDA ${CUDA_VER} selected and persisted.${NC}"
}

# Quick helper: ensure a valid CUDA is selected and persist it if missing from PATH.
# If PATH doesn't contain /usr/local/cuda/bin or nvcc not resolving there, offer to persist.
ensure_cuda_in_path_and_persist() {
    # run detection/selection if nvcc not in PATH
    if ! command -v nvcc &>/dev/null; then
        warn "${YELLOW}nvcc not found in PATH. Running detection...${NC}"
        detect_cuda
        return $?
    fi

    # check nvcc path
    nvcc_path="$(command -v nvcc)"
    nvcc_real="$(readlink -f "$nvcc_path" 2>/dev/null || true)"

    if [ -L /usr/local/cuda ]; then
        cur_target="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
        if [[ "$nvcc_real" == "$cur_target"* ]]; then
            info "${GREEN}nvcc already resolves to /usr/local/cuda installation.${NC}"
            return 0
        fi
    fi

    info "${YELLOW}nvcc found at: $nvcc_path${NC}"
    read -rp "Set /usr/local/cuda to this nvcc's parent and persist env? [y/N]: " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
        cuda_root="$(dirname "$(dirname "$nvcc_real")")"
        echo "Linking /usr/local/cuda -> $cuda_root"
        sudo ln -sfn "$cuda_root" /usr/local/cuda
        set_cuda_env_persistent "$cuda_root"
        info "${GREEN}Done. Re-login for persistent env to take effect.${NC}"
    else
        echo "No changes made."
    fi
}

set_cuda_env_persistent() {
    local target="${1:-/usr/local/cuda}"
    local content="# CUDA environment - auto-generated
export CUDA_PATH=\"$target\"
export PATH=\"\$CUDA_PATH/bin:\$PATH\"
export LD_LIBRARY_PATH=\"\$CUDA_PATH/lib64:\${LD_LIBRARY_PATH:-}\"
"

    # Always write to the user's shell rc (~/.bashrc) to persist for normal user
    local rc="$HOME/.bashrc"

    # Avoid duplicate append
    if ! grep -q "## CUDA environment - auto-generated" "$rc" 2>/dev/null; then
        printf "\n## CUDA environment - auto-generated\n" >> "$rc"
        printf '%s\n' "$content" >> "$rc"
        info "${GREEN}Appended CUDA env to $rc (applies at next login).${NC}"
    fi

    # Apply immediately to current shell (works for normal user)
    export CUDA_PATH="$target"
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
    update_torch_index_url
    info "${GREEN}CUDA environment variables set in current shell.${NC}"
}

# Interactively select CUDA (if >1), default to highest when Enter pressed
# After selection it sets CUDA_PATH, CUDA_VER (exported) and optionally makes /usr/local/cuda link and persists env
detect_cuda_select() {
    # Detect CUDA installations on disk
    detect_cuda_list
    _build_version_map

    # If CUDA already persisted and valid, skip selection
    if check_existing_cuda; then
        info "${GREEN}Existing CUDA installation detected; skipping selection.${NC}"
        return 0
    fi

    if [ "${#CUDA_MAP[@]}" -eq 0 ]; then
        error "${RED}No CUDA installations found on disk.${NC}"
        CUDA_VER=""
        CUDA_PATH=""
        return 1
    fi

    # Sort by version (semantic sort)
    IFS=$'\n' sorted=($(printf '%s\n' "${CUDA_MAP[@]}" | sort -t'|' -k1,1 -V))
    unset IFS
    CUDA_MAP_SORTED=("${sorted[@]}")

    # Determine default selection (highest version)
    default_index="${#CUDA_MAP_SORTED[@]}"
    highest="${CUDA_MAP_SORTED[-1]}"
    highest_ver="${highest%%|*}"
    highest_path="${highest#*|}"

    # Display available installations
    info "${GREEN}Detected CUDA installations:${NC}"
    local i=1
    for entry in "${CUDA_MAP_SORTED[@]}"; do
        ver="${entry%%|*}"
        path="${entry#*|}"
        marker=""
        if [ "$i" -eq "$default_index" ]; then
            marker="(default)"
        fi
        printf "  %2d) %s -> %s %s\n" "$i" "$ver" "$path" "$marker"
        ((i++))
    done

    # Prompt user to select
    read -rp "Select CUDA version [default: highest]: " choice
    if [ -z "$choice" ]; then
        choice_index=$default_index
    elif ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#CUDA_MAP_SORTED[@]}" ]; then
        warn "${YELLOW}Invalid choice, using default.${NC}"
        choice_index=$default_index
    else
        choice_index=$choice
    fi

    # Set selection
    selected="${CUDA_MAP_SORTED[$((choice_index-1))]}"
    CUDA_VER="${selected%%|*}"
    CUDA_PATH="${selected#*|}"

    echo "Linking /usr/local/cuda -> $CUDA_PATH"
    # sudo ln -sfn "$CUDA_PATH" /usr/local/cuda

    export CUDA_PATH
    export CUDA_VER
	export PATH="$CUDA_PATH/bin:$PATH"
	export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
    update_torch_index_url

    # Persist environment
    set_cuda_env_persistent "$CUDA_PATH"

    info "${GREEN}CUDA ${CUDA_VER} selected and persisted.${NC}"
}



# ------------------------------
# Show nvcc Version
# ------------------------------
show_nvcc_version() {
	auto_detect_nvcc
    if ! command -v nvcc &>/dev/null; then
        error "${RED}CUDA compiler (nvcc) not found.${NC}"
        return 1
    fi
    info "${GREEN}nvcc / CUDA version:${NC}"
    nvcc --version
}

auto_detect_nvcc() {
    if command -v nvcc &>/dev/null; then
        nvcc_dir="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
        echo "nvcc detected at $nvcc_dir"

        # If CUDA already set and matches detected nvcc, skip prompting
        if [ -n "${CUDA_PATH:-}" ] && [ "$(readlink -f "$CUDA_PATH")" = "$nvcc_dir" ]; then
            info "${GREEN}Detected nvcc matches current CUDA_PATH (${CUDA_PATH}). No change.${NC}"
            return 0
        fi

        read -rp "Use this CUDA and persist env? [y/N]: " ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            CUDA_PATH="$nvcc_dir"
            CUDA_VER="$("$CUDA_PATH/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
            # sudo ln -sfn "$CUDA_PATH" /usr/local/cuda
            set_cuda_env_persistent "$CUDA_PATH"
            info "${GREEN}CUDA ${CUDA_VER} persisted.${NC}"
        fi
    else
        info "${YELLOW}nvcc not found in PATH.${NC}"
    fi
}
