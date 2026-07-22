#!/usr/bin/env bash
# ============================================================
#  Bill OCR Pipeline - Environment Setup (Linux, PyTorch template)
#  Assumes Python/CUDA/PyTorch already provided by the base image.
#   - Ollama auto-detects GPU/CPU (no config needed).
#   - PaddlePaddle needs a different build per hardware.
# ============================================================

set -uo pipefail

error_exit() {
    echo
    echo "Setup failed - see errors above."
    exit 1
}

echo
echo "=== Checking for python3 / pip ==="
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH."
    exit 1
fi
python3 -m pip --version >/dev/null 2>&1 || python3 -m ensurepip --upgrade

echo
echo "=== Upgrading pip ==="
python3 -m pip install --upgrade pip

# ---- Ask which PaddlePaddle build to install ----
echo
echo "Which hardware should the OCR (PaddlePaddle) stage target?"
echo "  [1] CPU only        (works on every machine - default)"
echo "  [2] NVIDIA GPU CUDA (requires a supported NVIDIA GPU + drivers)"
read -r -p "Enter 1 or 2 [1]: " HWCHOICE
HWCHOICE="${HWCHOICE:-1}"

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
    fi
fi

echo
echo "=== Installing PDF / general dependencies ==="
python3 -m pip install pymupdf numpy pillow tqdm || error_exit

if [ "$HWCHOICE" = "2" ]; then
    echo
    echo "=== Installing PaddlePaddle GPU build (CUDA) ==="
    echo "NOTE: This uses PaddlePaddle's official CUDA wheel index."
    echo "      If your CUDA version differs, see https://www.paddlepaddle.org.cn/en/install/quick"
    if ! python3 -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/; then
        echo
        echo "GPU PaddlePaddle install failed. Falling back to CPU build..."
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

echo
echo "=== Installing Ollama Python client ==="
python3 -m pip install ollama || error_exit

echo
echo "=== Checking for Ollama runtime ==="
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
    echo "PaddleOCR: GPU (CUDA) build installed."
else
    echo "PaddleOCR: CPU build installed."
fi
echo "Ollama: runs on GPU automatically when available, CPU otherwise."
echo
echo "Run OCR stage:     python3 paddle_ocr_extract_pdf.py"
echo "Run JSON stage:    python3 03_run_llama_json.py ocr_out ocr_json_out"
echo
echo "(Ollama server was started in the background if it wasn't already running."
echo " Check /tmp/ollama.log if the JSON stage can't reach it.)"