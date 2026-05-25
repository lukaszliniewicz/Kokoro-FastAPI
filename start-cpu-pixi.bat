@echo off
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

if not defined PIXI_EXE set "PIXI_EXE=%USERPROFILE%\AppData\Local\Temp\opencode\tools\pixi.exe"
if not exist "%PIXI_EXE%" (
    where pixi >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Pixi executable not found.
        echo Set PIXI_EXE or install pixi in PATH.
        exit /b 1
    )
    set "PIXI_EXE=pixi"
)

if not defined PIXI_MANIFEST set "PIXI_MANIFEST=%USERPROFILE%\AppData\Local\Temp\opencode\pixi-kokoro-fastapi-cpu"
if not exist "%PIXI_MANIFEST%\pixi.toml" (
    echo [ERROR] Pixi manifest not found at "%PIXI_MANIFEST%\pixi.toml"
    echo Set PIXI_MANIFEST to your pixi workspace path.
    exit /b 1
)

set "MODEL_DIR=src/models"
set "VOICES_DIR=src/voices/v1_0"
set "USE_GPU=false"
set "PYTHONPATH=%REPO_ROOT%;%REPO_ROOT%\api"
set "PYTHONUTF8=1"
set "HF_HOME=%REPO_ROOT%\.hf-cache"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
if not defined DOWNLOAD_EXTERNAL_MODELS set "DOWNLOAD_EXTERNAL_MODELS=false"
if not defined DOWNLOAD_EXTERNAL_PROFILE_IDS set "DOWNLOAD_EXTERNAL_PROFILE_IDS="
if not defined DOWNLOAD_GERMAN_MARTIN set "DOWNLOAD_GERMAN_MARTIN=false"

if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%HUGGINGFACE_HUB_CACHE%" mkdir "%HUGGINGFACE_HUB_CACHE%"

set "BASE_MODEL_DIR=%REPO_ROOT%\api\src\models\v1_0"
set "BASE_MODEL_PATH=%BASE_MODEL_DIR%\kokoro-v1_0.pth"
set "GERMAN_MODEL_PATH=%REPO_ROOT%\api\src\models\kikiri_german_martin\kikiri_german_martin_ep10.pth"
set "GERMAN_VOICE_PATH=%REPO_ROOT%\api\src\voices\v1_0\martin.pt"
set "NEED_DOWNLOAD=false"

if not exist "%BASE_MODEL_PATH%" set "NEED_DOWNLOAD=true"

if /I "%DOWNLOAD_GERMAN_MARTIN%"=="true" (
    if not exist "%GERMAN_MODEL_PATH%" set "NEED_DOWNLOAD=true"
    if not exist "%GERMAN_VOICE_PATH%" set "NEED_DOWNLOAD=true"
)

if /I "%DOWNLOAD_EXTERNAL_MODELS%"=="true" set "NEED_DOWNLOAD=true"
if not "%DOWNLOAD_EXTERNAL_PROFILE_IDS%"=="" set "NEED_DOWNLOAD=true"

if /I "%NEED_DOWNLOAD%"=="true" (
    echo Required model assets are missing. Downloading...
    if not exist "%BASE_MODEL_DIR%" mkdir "%BASE_MODEL_DIR%"

    if /I "%DOWNLOAD_EXTERNAL_MODELS%"=="true" (
        if not "%DOWNLOAD_EXTERNAL_PROFILE_IDS%"=="" (
            if /I "%DOWNLOAD_GERMAN_MARTIN%"=="true" (
                "%PIXI_EXE%" run --manifest-path "%PIXI_MANIFEST%" python "%REPO_ROOT%\docker\scripts\download_model.py" --output "%BASE_MODEL_DIR%" --with-external-profiles --external-profile-ids "%DOWNLOAD_EXTERNAL_PROFILE_IDS%" --with-german-martin
            ) else (
                "%PIXI_EXE%" run --manifest-path "%PIXI_MANIFEST%" python "%REPO_ROOT%\docker\scripts\download_model.py" --output "%BASE_MODEL_DIR%" --with-external-profiles --external-profile-ids "%DOWNLOAD_EXTERNAL_PROFILE_IDS%"
            )
        ) else (
            if /I "%DOWNLOAD_GERMAN_MARTIN%"=="true" (
                "%PIXI_EXE%" run --manifest-path "%PIXI_MANIFEST%" python "%REPO_ROOT%\docker\scripts\download_model.py" --output "%BASE_MODEL_DIR%" --with-external-profiles --with-german-martin
            ) else (
                "%PIXI_EXE%" run --manifest-path "%PIXI_MANIFEST%" python "%REPO_ROOT%\docker\scripts\download_model.py" --output "%BASE_MODEL_DIR%" --with-external-profiles
            )
        )
    ) else (
        if not "%DOWNLOAD_EXTERNAL_PROFILE_IDS%"=="" (
            if /I "%DOWNLOAD_GERMAN_MARTIN%"=="true" (
                "%PIXI_EXE%" run --manifest-path "%PIXI_MANIFEST%" python "%REPO_ROOT%\docker\scripts\download_model.py" --output "%BASE_MODEL_DIR%" --with-external-profiles --external-profile-ids "%DOWNLOAD_EXTERNAL_PROFILE_IDS%" --with-german-martin
            ) else (
                "%PIXI_EXE%" run --manifest-path "%PIXI_MANIFEST%" python "%REPO_ROOT%\docker\scripts\download_model.py" --output "%BASE_MODEL_DIR%" --with-external-profiles --external-profile-ids "%DOWNLOAD_EXTERNAL_PROFILE_IDS%"
            )
        ) else (
            if /I "%DOWNLOAD_GERMAN_MARTIN%"=="true" (
                "%PIXI_EXE%" run --manifest-path "%PIXI_MANIFEST%" python "%REPO_ROOT%\docker\scripts\download_model.py" --output "%BASE_MODEL_DIR%" --with-german-martin
            ) else (
                "%PIXI_EXE%" run --manifest-path "%PIXI_MANIFEST%" python "%REPO_ROOT%\docker\scripts\download_model.py" --output "%BASE_MODEL_DIR%"
            )
        )
    )

    if errorlevel 1 (
        echo [ERROR] Failed to download required model assets.
        exit /b 1
    )
)

echo Starting Kokoro-FastAPI in pixi env on port 8880...
"%PIXI_EXE%" run --manifest-path "%PIXI_MANIFEST%" python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880

exit /b %errorlevel%
