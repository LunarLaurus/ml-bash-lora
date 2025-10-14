# conda-menu.sh
#!/bin/bash
source "$PROJECT_ROOT/helpers.sh"
update_script_dir 2

source "$SCRIPT_DIR/install_conda.sh"
source "$SCRIPT_DIR/env_manager.sh"

# ------------------------------
# Conda / ML Environment Menu
# Grouping: 0s general, 10s env creation, 20s installs, 30s utilities
# ------------------------------

env_disk_usage() {
    info "${GREEN}Disk usage of Conda environments:${NC}"
    du -sh $HOME/miniforge/envs/* 2>/dev/null || warn "${YELLOW}No conda envs found or du failed.${NC}"
}

ensure_python_cmd || { warn "${RED}Python not found for active environment. Activate/Create an Env. first.${NC}"; }

while true; do
    echo -e "\n${BCYAN}=== Conda / ML Environment Menu ===${NC}"
    check_env || true
    echo "1) Install Miniforge (Conda)"
    echo
    echo "10) Environment creation / activation"
    echo "    11) Create environment (prompt)"
    echo "    12) Activate environment (set PYTHON_CMD/PIP_CMD & save)"
    echo "    13) Run full setup (create+activate+installs)"
    echo
    echo "20) Install steps (run individually)"
    echo "    21) Detect GPU/CUDA"
    echo "    22) Install/ensure PyTorch"
    echo "    xx) Install LoRA stack (transformers, peft, datasets, accelerate, bitsandbytes)"
    echo "    24) Install RAG stack (faiss, sentence-transformers, langchain)"
    echo "    25) Validate environment"
    echo "    26) List packages in current environment"
    echo "    27) Remove package(s) from current environment"
    echo "    28) List LoRA lib versions (in active conda env)"
    echo "    29) List RAG lib versions (in active conda env)"
    echo
    echo "30) Management"
    echo "    31) Switch active ML environment"
    echo "    32) Remove ML environment"
    echo "    33) Show environment python version"
    echo "    34) Show Conda env disk usage"
    echo
    echo "0X) Quick full run"
    echo "    40) Run full setup (create+activate+all installs+validate)"
    echo
    echo -e "${BRED}0) Back to Main Menu${NC}"
    
    read -rp "Choice: " choice
    case $choice in
        1) install_conda ;;
        11) prompt_env_details; create_env ;;
        12) activate_env ;;
        13) run_full_setup ;;
        21) setup_gpu_cuda ;;
        22) install_pytorch_if_missing ;;
        # 23) install_lora_stack ;;
        24) install_rag_stack ;;
        25) validate_env ;;
        26) list_installed_packages ;;
        27) remove_user_packages ;;
        28) list_lora_lib_versions ;;
        29) list_rag_lib_versions ;;
        31) switch_env ;;
        32) remove_ml_env ;;
        33) show_python_version ;;
        34) env_disk_usage ;;
        40) run_full_setup ;;
        0|exit) break ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
