import json
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

import torch
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


def _read_version() -> str:
    version_file = Path(__file__).resolve().parents[3] / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    try:
        return _pkg_version("kokoro-fastapi")
    except PackageNotFoundError:
        return "0.0.0"


class ExternalModelProfile(BaseModel):
    """Config schema for optional externally hosted model profiles."""

    enabled: bool = True
    profile_id: str
    model_repo_id: str
    model_filename: str
    model_subdir: str
    model_ids: list[str] = Field(default_factory=list)
    lang_codes: list[str] = Field(default_factory=list)
    voice_files: dict[str, str] = Field(default_factory=dict)
    voice_names: list[str] = Field(default_factory=list)
    voice_aliases: dict[str, list[str]] = Field(default_factory=dict)
    voice_repo_id: str | None = None
    config_repo_id: str = "hexgrad/Kokoro-82M"
    config_filename: str = "config.json"


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Kokoro TTS API"
    api_description: str = "API for text-to-speech generation using Kokoro"
    api_version: str = _read_version()
    host: str = "0.0.0.0"
    port: int = 8880

    # Application Settings
    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0  # Maximum size of output directory in MB
    default_voice: str = "af_heart"
    default_voice_code: str | None = (
        None  # If set, overrides the first letter of voice name, though api call param still takes precedence
    )
    use_gpu: bool = True  # Whether to use GPU acceleration if available
    device_type: str | None = (
        None  # Will be auto-detected if None, can be "cuda", "mps", or "cpu"
    )
    allow_local_voice_saving: bool = (
        False  # Whether to allow saving combined voices locally
    )

    # Container absolute paths
    model_dir: str = "/app/api/src/models"  # Absolute path in container
    voices_dir: str = "/app/api/src/voices/v1_0"  # Absolute path in container

    # Dynamic asset loading
    enable_german_martin_support: bool = True
    auto_download_model_assets: bool = True

    # Generic external model profiles (JSON via EXTERNAL_MODEL_PROFILES)
    # Example value:
    # [{
    #   "profile_id": "kikiri-german-martin",
    #   "model_repo_id": "kikiri-tts/kikiri-german-martin",
    #   "model_filename": "kikiri_german_martin_ep10.pth",
    #   "model_subdir": "kikiri_german_martin",
    #   "model_ids": ["kikiri-german-martin"],
    #   "lang_codes": ["de", "d"],
    #   "voice_files": {"martin": "voices/martin.pt"},
    #   "voice_aliases": {"martin": ["martin"]},
    #   "config_repo_id": "hexgrad/Kokoro-82M",
    #   "config_filename": "config.json"
    # }]
    external_model_profiles: list[ExternalModelProfile] = Field(
        default_factory=list
    )

    # Legacy single-profile settings (still supported; mapped to one profile)
    german_model_repo_id: str = "kikiri-tts/kikiri-german-martin"
    german_model_filename: str = "kikiri_german_martin_ep10.pth"
    german_model_subdir: str = "kikiri_german_martin"
    german_voice_name: str = "martin"
    german_voice_aliases: list[str] = ["martin"]
    german_voice_filename_in_repo: str = "voices/martin.pt"
    german_config_repo_id: str = "hexgrad/Kokoro-82M"
    german_config_filename: str = "config.json"
    german_model_ids: list[str] = ["kikiri-german-martin"]

    # Audio Settings
    sample_rate: int = 24000
    default_volume_multiplier: float = 1.0
    # Text Processing Settings
    target_min_tokens: int = 175  # Target minimum tokens per chunk
    target_max_tokens: int = 250  # Target maximum tokens per chunk
    absolute_max_tokens: int = 450  # Absolute maximum tokens per chunk
    advanced_text_normalization: bool = True  # Preproesses the text before misiki
    voice_weight_normalization: bool = (
        True  # Normalize the voice weights so they add up to 1
    )

    gap_trim_ms: int = (
        1  # Base amount to trim from streaming chunk ends in milliseconds
    )
    dynamic_gap_trim_padding_ms: int = 410  # Padding to add to dynamic gap trim
    dynamic_gap_trim_padding_char_multiplier: dict[str, float] = {
        ".": 1,
        "!": 0.9,
        "?": 1,
        ",": 0.8,
    }

    # Web Player Settings
    enable_web_player: bool = True  # Whether to serve the web player UI
    web_player_path: str = "web"  # Path to web player static files
    cors_origins: list[str] = ["*"]  # CORS origins for web player
    cors_enabled: bool = True  # Whether to enable CORS

    # Temp File Settings for WEB Ui
    temp_file_dir: str = "api/temp_files"  # Directory for temporary audio files (relative to project root)
    max_temp_dir_size_mb: int = 2048  # Maximum size of temp directory (2GB)
    max_temp_dir_age_hours: int = 1  # Remove temp files older than 1 hour
    max_temp_dir_count: int = 3  # Maximum number of temp files to keep

    class Config:
        env_file = ".env"

    @field_validator("external_model_profiles", mode="before")
    @classmethod
    def _coerce_external_model_profiles(cls, value):
        if value in (None, ""):
            return []

        if isinstance(value, dict):
            return [value]

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return value

            if isinstance(parsed, dict):
                return [parsed]
            return parsed

        return value

    def get_device(self) -> str:
        """Get the appropriate device based on settings and availability"""
        if not self.use_gpu:
            return "cpu"

        if self.device_type:
            return self.device_type

        # Auto-detect device
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"


settings = Settings()
