# ML/LoRA + RAG Setup and Management System

This project is a modular, menu-driven setup and management system for ML/LoRA and RAG workflows on Ubuntu 22/24 LTS with NVIDIA GPUs. The scripts are split into multiple files with separate menus and submenus, making it easier to manage drivers, CUDA, Conda environments, PyTorch, and system diagnostics.

## Key Features

- Main Menu provides access to NVIDIA drivers, CUDA toolkit, Conda/ML environments, PyTorch installation, and diagnostics  
- NVIDIA Drivers Menu allows installing/updating NVIDIA drivers and checking GPU status via nvidia-smi  
- CUDA Toolkit Menu supports detecting CUDA installations, installing new CUDA versions, listing available versions from apt, removing specific or obsolete CUDA versions, showing nvcc version, and auto-detecting nvcc  
- Conda / ML Environment Menu enables installing Miniforge (Conda), creating new ML environments, switching between environments, removing environments, and checking Python version and disk usage in environments  
- PyTorch Installation Menu installs or upgrades PyTorch matching the detected CUDA version, with automatic wheel selection and validation  
- Diagnostics Menu provides quick checks for disk usage, Python version, and Conda environment disk usage  
- Helper scripts (helpers.sh, install_conda.sh, env_manager.sh, detect_cuda.sh, install_cuda.sh, remove_cuda.sh, install_pytorch.sh) centralize functionality like environment detection, CUDA handling, PyTorch installation, and system utilities  

## Environment and CUDA Management

- Tracks the active ML environment and persists it across sessions  
- Detects and maps CUDA installations on disk, allows interactive selection, and sets /usr/local/cuda symlink  
- Persists CUDA environment variables in .bashrc for future sessions  
- Supports removal of specific or obsolete CUDA versions while keeping the latest  

## PyTorch and ML Stack

- Installs PyTorch wheels compatible with the detected CUDA version  
- Installs LoRA/Hugging Face stack: transformers, peft, datasets, accelerate, bitsandbytes  
- Optionally installs RAG stack: faiss-cpu, sentence-transformers, langchain  
- Validates GPU, CUDA, PyTorch, and bitsandbytes installation in the environment  

## System Utilities

- Shows disk usage, Python version, Conda environment sizes, and GPU status  
- Menu-driven design ensures tasks can be performed interactively with clear prompts  

## Menus and Submenus

- Main Menu: NVIDIA Drivers, CUDA Toolkit, Conda/ML Environments, PyTorch Installation, Diagnostics, Exit  
- NVIDIA Drivers Menu: Install/Update drivers, Show nvidia-smi, Back  
- CUDA Toolkit Menu: Detect/Select CUDA, Install CUDA, List available CUDA versions, Remove CUDA versions, Remove obsolete CUDA versions, Show nvcc, Auto Detect nvcc, Back  
- Conda / ML Environment Menu: Install Conda, Create environment, Switch environment, Remove environment, Show Python version, Show disk usage, Back  
- PyTorch Installation Menu: Install/Upgrade PyTorch, Back  
- Diagnostics Menu: Disk Usage, Python version, Conda environment disk usage, Back  

