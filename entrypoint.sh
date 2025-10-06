#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Source helpers first
source "$(dirname "$0")/helpers/helpers.sh"

echo "=============================="
echo " ML/LoRA + RAG Setup Script"
echo " Auto-detect GPU/CUDA/PyTorch"
echo " Ubuntu 22.04 / RTX 4000+"
echo "=============================="

bash "$PROJECT_ROOT/main-menu.sh"