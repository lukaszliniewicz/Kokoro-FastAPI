$env:PHONEMIZER_ESPEAK_LIBRARY="C:\Program Files\eSpeak NG\libespeak-ng.dll"
$env:PYTHONUTF8=1
$Env:PROJECT_ROOT="$pwd"
$Env:USE_GPU="true"
$Env:USE_ONNX="false"
$Env:PYTHONPATH="$Env:PROJECT_ROOT;$Env:PROJECT_ROOT/api"
$Env:MODEL_DIR="src/models"
$Env:VOICES_DIR="src/voices/v1_0"
$Env:WEB_PLAYER_PATH="$Env:PROJECT_ROOT/web"

if (-not $Env:DOWNLOAD_GERMAN_MARTIN) {
    $Env:DOWNLOAD_GERMAN_MARTIN = "false"
}
if (-not $Env:DOWNLOAD_EXTERNAL_MODELS) {
    $Env:DOWNLOAD_EXTERNAL_MODELS = "false"
}
if (-not $Env:DOWNLOAD_EXTERNAL_PROFILE_IDS) {
    $Env:DOWNLOAD_EXTERNAL_PROFILE_IDS = ""
}

uv pip install -e ".[gpu]"
$downloadArgs = @("--output", "api/src/models/v1_0")
if ($Env:DOWNLOAD_EXTERNAL_MODELS.ToLower() -eq "true") {
    $downloadArgs += "--with-external-profiles"
}
if (-not [string]::IsNullOrWhiteSpace($Env:DOWNLOAD_EXTERNAL_PROFILE_IDS)) {
    $downloadArgs += @("--with-external-profiles", "--external-profile-ids", $Env:DOWNLOAD_EXTERNAL_PROFILE_IDS)
}
if ($Env:DOWNLOAD_GERMAN_MARTIN.ToLower() -eq "true") {
    $downloadArgs += "--with-german-martin"
}
uv run --no-sync python docker/scripts/download_model.py @downloadArgs
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880
