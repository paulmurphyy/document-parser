@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Bill OCR Pipeline - Environment Setup (Windows)
REM  Supports BOTH CPU-only and NVIDIA GPU machines.
REM   - Ollama auto-detects GPU/CPU (no config needed).
REM   - PaddlePaddle needs a different build per hardware.
REM ============================================================

echo.
echo === Checking for conda ===
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda not found on PATH.
    echo Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM ---- Ask which PaddlePaddle build to install ----
echo.
echo Which hardware should the OCR (PaddlePaddle) stage target?
echo   [1] CPU only        (works on every machine - default)
echo   [2] NVIDIA GPU CUDA (requires a supported NVIDIA GPU + drivers)
set "HWCHOICE="
set /p HWCHOICE="Enter 1 or 2 [1]: "
if "!HWCHOICE!"=="" set "HWCHOICE=1"

if "!HWCHOICE!"=="2" (
    echo.
    echo === Verifying NVIDIA GPU is visible (nvidia-smi) ===
    where nvidia-smi >nul 2>nul
    if errorlevel 1 (
        echo WARNING: nvidia-smi not found. No usable NVIDIA GPU detected.
        echo Falling back to the CPU build to avoid a broken install.
        set "HWCHOICE=1"
    ) else (
        nvidia-smi >nul 2>nul
        if errorlevel 1 (
            echo WARNING: nvidia-smi failed. Falling back to CPU build.
            set "HWCHOICE=1"
        )
    )
)

echo.
echo === Creating conda environment "bill-pipeline" (Python 3.11) ===
call conda create -y -n bill-pipeline python=3.11
if errorlevel 1 goto :error
call conda activate bill-pipeline
if errorlevel 1 goto :error

echo.
echo === Upgrading pip ===
python -m pip install --upgrade pip

echo.
echo === Installing PDF / general dependencies ===
pip install pymupdf numpy pillow tqdm
if errorlevel 1 goto :error

if "!HWCHOICE!"=="2" (
    echo.
    echo === Installing PaddlePaddle GPU build ^(CUDA^) ===
    echo NOTE: This uses PaddlePaddle's official CUDA wheel index.
    echo       If your CUDA version differs, see https://www.paddlepaddle.org.cn/en/install/quick
    pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    if errorlevel 1 (
        echo.
        echo GPU PaddlePaddle install failed. Falling back to CPU build...
        pip install paddlepaddle==3.2.1
        if errorlevel 1 goto :error
    )
) else (
    echo.
    echo === Installing PaddlePaddle CPU build ^(pinned 3.2.1 - do not bump^) ===
    pip install paddlepaddle==3.2.1
    if errorlevel 1 goto :error
)

echo.
echo === Installing PaddleOCR + PaddleX (pinned) ===
pip install paddleocr==3.7.0 "paddlex[ocr]==3.7.2"
if errorlevel 1 goto :error

echo.
echo === Installing Ollama Python client ===
pip install ollama
if errorlevel 1 goto :error

echo.
echo === Checking for Ollama runtime ===
where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama not found on PATH. Attempting install via winget...
    winget install -e --id Ollama.Ollama
    if errorlevel 1 (
        echo.
        echo Could not install Ollama automatically.
        echo Install manually from https://ollama.com/download then re-run this script.
        exit /b 1
    )
    echo.
    echo NOTE: Open a NEW terminal after install so "ollama" is on PATH,
    echo then re-run this script to finish pulling the model.
    goto :eof
)

echo.
echo === Pulling model (Ollama auto-uses GPU if present, else CPU) ===
ollama pull granite4.1:3b
if errorlevel 1 goto :error

echo.
echo === Setup complete ===
if "!HWCHOICE!"=="2" (
    echo PaddleOCR: GPU ^(CUDA^) build installed.
) else (
    echo PaddleOCR: CPU build installed.
)
echo Ollama: runs on GPU automatically when available, CPU otherwise.
echo.
echo Activate with:     conda activate bill-pipeline
echo Run OCR stage:     python paddle_ocr_extract_pdf.py
echo Run JSON stage:    python 03_run_llama_json.py ocr_out ocr_json_out
echo.
echo (Ensure Ollama is running before the JSON stage: it starts after install,
echo  or run "ollama serve" manually.)
goto :eof

:error
echo.
echo Setup failed - see errors above.
exit /b 1
