#!/bin/bash
# git helper functions — all repo operations use $REPO_BASE_DIR/<repo-folder>

: "${PROJECT_ROOT:=${PWD}}"
REPO_BASE_DIR="${PROJECT_ROOT}/repo-data"
[ ! -d "$REPO_BASE_DIR" ] && mkdir -p "$REPO_BASE_DIR"

# Helper: get local folder name from .git URL
repo_folder_from_url() {
    local url="$1"
    local name="${url##*/}"
    name="${name%.git}"
    printf "%s" "$name"
}

# Resolve a selection (index or folder) into URL, folder and local_path
# Globals set:
#   REPO_SEL_URL, REPO_SEL_FOLDER, REPO_LOCAL_PATH
resolve_selection() {
    local sel="$1"
    REPO_SEL_URL=""
    REPO_SEL_FOLDER=""
    REPO_LOCAL_PATH=""
    
    if [[ -z "${sel:-}" ]]; then
        return 1
    fi
    
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        if (( sel < 0 || sel >= ${#poke_repos_mainline[@]} )); then
            return 2
        fi
        REPO_SEL_URL="${poke_repos_mainline[$sel]}"
        REPO_SEL_FOLDER="$(repo_folder_from_url "$REPO_SEL_URL")"
    else
        # try match by folder name in poke_repos_mainline
        for u in "${poke_repos_mainline[@]}"; do
            if [ "$(repo_folder_from_url "$u")" = "$sel" ]; then
                REPO_SEL_URL="$u"
                REPO_SEL_FOLDER="$(repo_folder_from_url "$u")"
                break
            fi
        done
        # if no known URL matched, treat input as folder name
        if [ -z "$REPO_SEL_URL" ]; then
            REPO_SEL_FOLDER="$sel"
        fi
    fi
    
    REPO_LOCAL_PATH="$REPO_BASE_DIR/$REPO_SEL_FOLDER"
    return 0
}

# List available repos and whether they exist locally (under REPO_BASE_DIR)
list_repos() {
    echo -e "\n${BGREEN}Available repos:${NC}"
    local i=0
    for url in "${poke_repos_mainline[@]}"; do
        local folder local_path exists
        folder="$(repo_folder_from_url "$url")"
        local_path="$REPO_BASE_DIR/$folder"
        exists="no"
        if [ -d "$local_path/.git" ]; then
            exists="yes"
        fi
        printf " %2d) %s -> %s (local: %s)\n" "$i" "$folder" "$url" "$exists"
        i=$((i + 1))
    done
}

# Clone a single repo (by index or name) into REPO_BASE_DIR/<folder>
clone_repo() {
    list_repos
    read -rp "Enter repo index or folder name to clone (or 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    if ! resolve_selection "$sel"; then
        echo -e "${BRED}Selection invalid${NC}"
        return 1
    fi
    
    if [ -d "$REPO_LOCAL_PATH" ]; then
        echo -e "${BRED}Folder '$REPO_LOCAL_PATH' already exists. Skipping clone.${NC}"
        return 1
    fi
    
    echo -e "${BGREEN}Cloning ${REPO_SEL_URL:-<unknown-url>} into $REPO_LOCAL_PATH ...${NC}"
    git clone "${REPO_SEL_URL:-}" "$REPO_LOCAL_PATH" || {
        echo -e "${BRED}git clone failed for ${REPO_SEL_URL:-}${NC}"
        return 1
    }
    echo -e "${BGREEN}Clone finished: $REPO_LOCAL_PATH${NC}"
}

# Clone all repos in the array (skip existing)
clone_all_repos() {
    for url in "${poke_repos_mainline[@]}"; do
        local folder local_path
        folder="$(repo_folder_from_url "$url")"
        local_path="$REPO_BASE_DIR/$folder"
        if [ -d "$local_path/.git" ]; then
            echo -e "${BRED}Skipping $local_path (already exists)${NC}"
            continue
        fi
        echo -e "${BGREEN}Cloning $url into $local_path ...${NC}"
        git clone "$url" "$local_path" || {
            echo -e "${BRED}Failed to clone $url${NC}"
        }
    done
    echo -e "${BGREEN}Done cloning all repos.${NC}"
}

# Run git pull in a repo (or all)
update_repo() {
    list_repos
    read -rp "Enter repo index or folder name to update (or 'all' to update all, 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    if [ "$sel" = "all" ]; then
        for url in "${poke_repos_mainline[@]}"; do
            folder="$(repo_folder_from_url "$url")"
            local local_path="$REPO_BASE_DIR/$folder"
            if [ -d "$local_path/.git" ]; then
                echo -e "${BGREEN}Updating $folder ...${NC}"
                (cd "$local_path" && git pull --ff-only) || echo -e "${BRED}Failed to update $folder${NC}"
            else
                echo -e "${BRED}Skipping $folder (not cloned)${NC}"
            fi
        done
        return 0
    fi
    
    if ! resolve_selection "$sel"; then
        echo -e "${BRED}Invalid selection${NC}"
        return 1
    fi
    
    if [ ! -d "$REPO_LOCAL_PATH/.git" ]; then
        echo -e "${BRED}Repo '${REPO_SEL_FOLDER}' not cloned (missing at $REPO_LOCAL_PATH)${NC}"
        return 1
    fi
    
    echo -e "${BGREEN}Running git pull in $REPO_LOCAL_PATH ...${NC}"
    (cd "$REPO_LOCAL_PATH" && git pull --ff-only) || echo -e "${BRED}git pull failed for ${REPO_SEL_FOLDER}${NC}"
}

# Show git status for a repo
repo_status() {
    list_repos
    read -rp "Enter repo index or folder name to show status (or 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    if ! resolve_selection "$sel"; then
        echo -e "${BRED}Invalid selection${NC}"
        return 1
    fi
    
    if [ ! -d "$REPO_LOCAL_PATH/.git" ]; then
        echo -e "${BRED}Repo '${REPO_SEL_FOLDER}' not cloned (missing at $REPO_LOCAL_PATH)${NC}"
        return 1
    fi
    
    (cd "$REPO_LOCAL_PATH" && echo -e "${BGREEN}== ${REPO_SEL_FOLDER} status ==${NC}" && git status -s)
}

# Delete a repo's local folder (with confirmation)
delete_repo() {
    list_repos
    read -rp "Enter repo index or folder name to delete (local folder only) (or 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    if ! resolve_selection "$sel"; then
        echo -e "${BRED}Invalid selection${NC}"
        return 1
    fi
    
    if [ ! -d "$REPO_LOCAL_PATH" ]; then
        echo -e "${BRED}Folder '$REPO_LOCAL_PATH' does not exist locally.${NC}"
        return 1
    fi
    
    read -rp "Are you sure you want to DELETE the local folder '$REPO_LOCAL_PATH'? This cannot be undone. (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -rf -- "$REPO_LOCAL_PATH"
        echo -e "${BGREEN}Deleted $REPO_LOCAL_PATH${NC}"
    else
        echo -e "${BRED}Aborted${NC}"
    fi
}

# Open a subshell in the repo folder
open_repo_shell() {
    list_repos
    read -rp "Enter repo index or folder name to open a subshell in (or 'q' to cancel): " sel
    [ "$sel" = "q" ] && return 0
    
    if ! resolve_selection "$sel"; then
        echo -e "${BRED}Invalid selection${NC}"
        return 1
    fi
    
    if [ ! -d "$REPO_LOCAL_PATH" ]; then
        echo -e "${BRED}Folder '$REPO_LOCAL_PATH' does not exist.${NC}"
        return 1
    fi
    
    echo -e "${BGREEN}Spawning subshell in $REPO_LOCAL_PATH (type exit to return)...${NC}"
    pushd "$REPO_LOCAL_PATH" >/dev/null || return 1
    bash --login
    popd >/dev/null || true
}