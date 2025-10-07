#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
source "$PROJECT_ROOT/cuda/detect_cuda.sh"


remove_cuda_version() {
    # Detect installed CUDA versions
    detect_cuda_list
    _build_version_map

    if [ "${#CUDA_MAP[@]}" -eq 0 ]; then
        echo -e "${RED}No CUDA installations found on disk.${NC}"
        return 1
    fi

    # Sort by version
    IFS=$'\n' sorted=($(printf '%s\n' "${CUDA_MAP[@]}" | sort -t'|' -k1,1 -V))
    unset IFS
    CUDA_MAP_SORTED=("${sorted[@]}")

    # Show installations
    echo -e "${GREEN}Installed CUDA versions:${NC}"
    local i=1
    for entry in "${CUDA_MAP_SORTED[@]}"; do
        ver="${entry%%|*}"
        path="${entry#*|}"
        printf "  %2d) %s -> %s\n" "$i" "$ver" "$path"
        ((i++))
    done

    # Prompt user for selection
    read -rp "Enter number of CUDA version to remove (or multiple comma-separated): " choices
    IFS=',' read -ra choice_arr <<< "$choices"

    for c in "${choice_arr[@]}"; do
        if ! [[ "$c" =~ ^[0-9]+$ ]] || [ "$c" -lt 1 ] || [ "$c" -gt "${#CUDA_MAP_SORTED[@]}" ]; then
            echo -e "${YELLOW}Skipping invalid choice: $c${NC}"
            continue
        fi

        selected="${CUDA_MAP_SORTED[$((c-1))]}"
        ver="${selected%%|*}"
        path="${selected#*|}"

        echo -e "${YELLOW}Removing CUDA $ver at $path ...${NC}"
        sudo rm -rf "$path"

        # Check if /usr/local/cuda points here, remove link if so
        if [ -L /usr/local/cuda ]; then
            cur_link=$(readlink -f /usr/local/cuda)
            if [ "$cur_link" = "$path" ]; then
                echo "Removing /usr/local/cuda symlink pointing to removed version"
                sudo rm -f /usr/local/cuda
            fi
        fi
    done

    echo -e "${GREEN}CUDA removal complete.${NC}"
}
remove_obsolete_cuda_versions() {
    # Detect installed CUDA versions
    detect_cuda_list
    _build_version_map

    if [ "${#CUDA_MAP[@]}" -eq 0 ]; then
        echo -e "${RED}No CUDA installations found on disk.${NC}"
        return 1
    fi

    # Sort by version (highest last)
    IFS=$'\n' sorted=($(printf '%s\n' "${CUDA_MAP[@]}" | sort -t'|' -k1,1 -V))
    unset IFS
    CUDA_MAP_SORTED=("${sorted[@]}")

    # Keep latest version (highest)
    latest="${CUDA_MAP_SORTED[-1]}"
    latest_ver="${latest%%|*}"
    latest_path="${latest#*|}"

    echo -e "${GREEN}Latest CUDA version will be kept: ${latest_ver} -> ${latest_path}${NC}"

    # Remove all except latest
    for ((i=0; i<${#CUDA_MAP_SORTED[@]}-1; i++)); do
        entry="${CUDA_MAP_SORTED[$i]}"
        ver="${entry%%|*}"
        path="${entry#*|}"

        echo -e "${YELLOW}Removing obsolete CUDA $ver at $path ...${NC}"
        sudo rm -rf "$path"

        # Remove /usr/local/cuda symlink if it points to this version
        if [ -L /usr/local/cuda ]; then
            cur_link=$(readlink -f /usr/local/cuda)
            if [ "$cur_link" = "$path" ]; then
                echo "Removing /usr/local/cuda symlink pointing to removed version"
                sudo rm -f /usr/local/cuda
            fi
        fi
    done

    echo -e "${GREEN}Obsolete CUDA versions removed. Latest version ${latest_ver} retained.${NC}"
}