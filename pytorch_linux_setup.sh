#!/usr/bin/env bash
# ============================================================
#  Bill OCR Pipeline - Environment Setup (Linux, PyTorch template)
#  Assumes CUDA drivers are already provided by the base image.
#   - Installs into a persistent virtualenv (survives container
#     rebuilds) instead of the base image's system site-packages.
#   - This is what fixes the torch/paddle shared-CUDA-lib conflict:
#     paddlex pulls in modelscope -> torch as a side effect, and if
#     that lands in the SAME site-packages as your base-image torch,
#     pip's resolver fights over one shared nvidia-nccl-cu12 /
#     nvidia-cublas-cu12 / etc version for both packages. Installing
#     everything into a brand-new venv lets pip resolve a torch build
#     that's actually compatible with whatever paddle pins, instead of
#     clobbering a torch that was already installed for something else.
#   - Ollama auto-detects GPU/CPU (no config needed) and stays a
#     system-level binary; only its Python client goes in the venv.
# ============================================================
set -uo pipefail
error_exit() {
    echo
    echo "Setup failed - see errors above."
    exit 1
}

echo
echo "=== Checking for python3 ==="
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH."
    exit 1
fi

# ---- Persistent venv location ----
echo
echo "Where should the persistent virtualenv live?"
echo "(This must be on your persistent storage volume, not ephemeral"
echo " container storage, or you'll have to reinstall everything on restart.)"
read -r -p "Venv path [/workspace/persistent/paddle-ocr-venv]: " VENV_DIR
VENV_DIR="${VENV_DIR:-/workspace/persistent/paddle-ocr-venv}"

mkdir -p "$(dirname "$VENV_DIR")" || error_exit

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Found existing venv at $VENV_DIR - reusing it."
else
    echo
    echo "=== Creating virtualenv at $VENV_DIR ==="
    python3 -m venv "$VENV_DIR" || error_exit
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate" || error_exit
echo "Active Python: $(command -v python3)"

echo
echo "=== Upgrading pip inside venv ==="
python3 -m pip install --upgrade pip

# ---- Ask which PaddlePaddle build to install ----
echo
echo "Which hardware should the OCR (PaddlePaddle) stage target?"
echo "  [1] CPU only        (works on every machine - default)"
echo "  [2] NVIDIA GPU CUDA (requires a supported NVIDIA GPU + drivers)"
read -r -p "Enter 1 or 2 [1]: " HWCHOICE
HWCHOICE="${HWCHOICE:-1}"

CUDA_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"

if [ "$HWCHOICE" = "2" ]; then
    echo
    echo "=== Verifying NVIDIA GPU is visible (nvidia-smi) ==="
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "WARNING: nvidia-smi not found. No usable NVIDIA GPU detected."
        echo "Falling back to the CPU build to avoid a broken install."
        HWCHOICE=1
    elif ! nvidia-smi >/dev/null 2>&1; then
        echo "WARNING: nvidia-smi failed. Falling back to CPU build."
        HWCHOICE=1
    else
        echo
        echo "Detected driver / CUDA version:"
        nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true
        nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true
        echo
        echo "Default wheel index targets CUDA 12.6 (cu126), which is forward-"
        echo "compatible with newer driver-reported CUDA versions (e.g. 12.8)."
        read -r -p "Override CUDA index [cu126/cu118/cu129/cu130] (blank = cu126): " CUDA_TAG
        if [ -n "${CUDA_TAG:-}" ]; then
            CUDA_INDEX="https://www.paddlepaddle.org.cn/packages/stable/${CUDA_TAG}/"
        fi
    fi
fi

echo
echo "=== Installing PDF / general dependencies ==="
python3 -m pip install pymupdf numpy pillow tqdm || error_exit

if [ "$HWCHOICE" = "2" ]; then
    echo
    echo "=== Installing PaddlePaddle GPU build (CUDA) ==="
    echo "NOTE: Using index: $CUDA_INDEX"
    echo "      If this fails or your CUDA version differs, see"
    echo "      https://www.paddlepaddle.org.cn/en/install/quick"
    if ! python3 -m pip install paddlepaddle-gpu==3.2.1 -i "$CUDA_INDEX" --retries 10 --timeout 120; then
        echo
        echo "GPU PaddlePaddle install failed. Falling back to CPU build..."
        HWCHOICE=1
        python3 -m pip install paddlepaddle==3.2.1 || error_exit
    fi
else
    echo
    echo "=== Installing PaddlePaddle CPU build (pinned 3.2.1 - do not bump) ==="
    python3 -m pip install paddlepaddle==3.2.1 || error_exit
fi

echo
echo "=== Installing PaddleOCR + PaddleX (pinned) ==="
python3 -m pip install paddleocr==3.7.0 "paddlex[ocr]==3.7.2" || error_exit

if [ "$HWCHOICE" = "2" ]; then
    echo
    echo "=== Sanity-checking torch import (pulled in transitively via modelscope) ==="
    echo "    This is the step that broke before when torch/paddle shared one"
    echo "    site-packages. Inside this venv it should resolve cleanly on its own."
    if ! python3 -c "import torch; print('torch OK, cuda available:', torch.cuda.is_available())"; then
        echo
        echo "WARNING: torch import failed inside the venv. Common fix: force a"
        echo "reinstall of torch matching the nvidia-nccl-cu12 version paddle pinned:"
        python3 -m pip show nvidia-nccl-cu12 2>/dev/null | grep -i version || true
        echo "Try: pip install torch --index-url https://download.pytorch.org/whl/cu126"
    fi
    echo
    echo "=== Verifying PaddlePaddle sees the GPU ==="
    python3 -c "import paddle; print('paddle CUDA:', paddle.is_compiled_with_cuda(), '| devices:', paddle.device.cuda.device_count())" || true
fi

echo
echo "=== Installing Ollama Python client ==="
python3 -m pip install ollama || error_exit

echo
echo "=== Checking for Ollama runtime (system-level, not venv) ==="
if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not found on PATH. Attempting install via official install script..."
    if ! curl -fsSL https://ollama.com/install.sh | sh; then
        echo
        echo "Could not install Ollama automatically."
        echo "Install manually from https://ollama.com/download then re-run this script."
        exit 1
    fi
fi

echo
echo "=== Ensuring Ollama server is running ==="
if ! pgrep -x "ollama" >/dev/null 2>&1; then
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    sleep 3
fi

echo
echo "=== Pulling model (Ollama auto-uses GPU if present, else CPU) ==="
ollama pull granite4.1:3b || error_exit

echo
echo "=== Setup complete ==="
if [ "$HWCHOICE" = "2" ]; then
    echo "PaddleOCR: GPU (CUDA, $CUDA_INDEX) build installed."
else
    echo "PaddleOCR: CPU build installed."
fi
echo "Ollama: runs on GPU automatically when available, CPU otherwise."
echo "Venv: $VENV_DIR (persistent - reused on next run if it still exists)"
echo
echo "IMPORTANT: activate this venv before running the pipeline in new shells:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Run OCR stage:     python3 paddle_ocr_extract_pdf.py"
echo "Run JSON stage:    python3 03_run_llama_json.py ocr_out ocr_json_out"
echo
echo "(Ollama server was started in the background if it wasn't already running."
echo " Check /tmp/ollama.log if the JSON stage can't reach it.)"
