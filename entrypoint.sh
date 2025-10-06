#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# ------------------------------
# Script directory helpers
# ------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================="
echo " ML/LoRA + RAG Setup Script"
echo " Auto-detect GPU/CUDA/PyTorch"
echo " Ubuntu 22.04 / RTX 4000+"
echo "=============================="
echo "PROJECT_ROOT: $PROJECT_ROOT"

source "$PROJECT_ROOT/main-menu.sh"