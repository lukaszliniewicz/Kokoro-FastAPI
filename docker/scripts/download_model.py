#!/usr/bin/env python3
"""Download and prepare Kokoro model assets.

By default this script downloads the base Kokoro v1.0 model/config into
``--output``.

Optional external profile assets can also be downloaded.
Profiles are loaded from ``EXTERNAL_MODEL_PROFILES`` (JSON list).
If that env var is not set, the script falls back to legacy German Martin
settings for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve


BASE_RELEASE_URL = os.getenv(
    "KOKORO_BASE_RELEASE_URL",
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.4",
)
BASE_MODEL_FILE = "kokoro-v1_0.pth"
BASE_CONFIG_FILE = "config.json"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_token(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower()


def _strip_voice_extension(name: str) -> str:
    value = name.strip()
    if value.lower().endswith(".pt"):
        return value[:-3]
    return value


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _Logger:
    def info(self, msg: str) -> None:
        _log(msg)

    def warning(self, msg: str) -> None:
        _log(f"WARNING: {msg}")

    def error(self, msg: str) -> None:
        _log(f"ERROR: {msg}")


logger = _Logger()


@dataclass(frozen=True)
class ExternalProfile:
    profile_id: str
    model_repo_id: str
    model_filename: str
    model_subdir: str
    model_ids: tuple[str, ...]
    voice_files: dict[str, str]
    voice_repo_id: str | None
    config_repo_id: str
    config_filename: str
    enabled: bool = True


def _verify_json_file(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            return False
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def _verify_non_empty_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def verify_base_model_files(model_path: Path, config_path: Path) -> bool:
    """Verify that base model + config files exist and are valid."""
    return _verify_non_empty_file(model_path) and _verify_json_file(config_path)


def _download_http(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(output_path))


def _download_hf_file(repo_id: str, filename: str, output_path: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to download external profile assets"
        ) from exc

    source_path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)


def download_base_model(output_dir: str) -> None:
    """Download base Kokoro v1.0 model files from release artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / BASE_MODEL_FILE
    config_path = output_path / BASE_CONFIG_FILE

    if verify_base_model_files(model_path, config_path):
        logger.info("Base model files already exist and are valid")
        return

    logger.info("Downloading base Kokoro v1.0 model files")

    model_url = f"{BASE_RELEASE_URL}/{BASE_MODEL_FILE}"
    config_url = f"{BASE_RELEASE_URL}/{BASE_CONFIG_FILE}"

    logger.info("Downloading base model file...")
    _download_http(model_url, model_path)

    logger.info("Downloading base config file...")
    _download_http(config_url, config_path)

    if not verify_base_model_files(model_path, config_path):
        raise RuntimeError("Failed to verify downloaded base model files")

    logger.info(f"Base model files prepared in {output_path.as_posix()}")


def _resolve_models_root(base_output: Path, models_root: str | None) -> Path:
    if models_root:
        return Path(models_root).resolve()
    return base_output.parent.resolve()


def _resolve_voices_output(base_output: Path, voices_output: str | None) -> Path:
    if voices_output:
        return Path(voices_output).resolve()

    # Expected default layout from --output: api/src/models/v1_0
    # Derived voice path: api/src/voices/v1_0
    src_root = base_output.parent.parent.resolve()
    return src_root / "voices" / "v1_0"


def _normalize_voice_files(raw: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for voice_name, repo_filename in raw.items():
        canonical = _normalize_token(_strip_voice_extension(voice_name))
        repo_path = (repo_filename or "").strip()
        if canonical and repo_path:
            normalized[canonical] = repo_path
    return normalized


def _profile_from_dict(data: dict) -> ExternalProfile:
    model_repo_id = (data.get("model_repo_id") or "").strip()
    model_filename = (data.get("model_filename") or "").strip()
    model_subdir = (data.get("model_subdir") or "").strip()
    profile_id = (data.get("profile_id") or "").strip() or model_subdir

    if not profile_id or not model_repo_id or not model_filename:
        raise ValueError(
            "External profile requires profile_id, model_repo_id, model_filename"
        )

    voice_files_raw = data.get("voice_files") or {}
    if not isinstance(voice_files_raw, dict):
        raise ValueError("voice_files must be an object mapping local-name -> repo path")

    normalized_voice_files = _normalize_voice_files(voice_files_raw)

    model_ids_raw = data.get("model_ids") or []
    if not isinstance(model_ids_raw, list):
        raise ValueError("model_ids must be a list")
    model_ids = tuple(
        item.strip() for item in model_ids_raw if isinstance(item, str) and item.strip()
    )

    return ExternalProfile(
        profile_id=profile_id,
        model_repo_id=model_repo_id,
        model_filename=model_filename,
        model_subdir=model_subdir,
        model_ids=model_ids,
        voice_files=normalized_voice_files,
        voice_repo_id=(data.get("voice_repo_id") or None),
        config_repo_id=(data.get("config_repo_id") or "hexgrad/Kokoro-82M").strip(),
        config_filename=(data.get("config_filename") or "config.json").strip(),
        enabled=bool(data.get("enabled", True)),
    )


def _legacy_german_profile() -> ExternalProfile | None:
    if not _env_flag("ENABLE_GERMAN_MARTIN_SUPPORT", True):
        return None

    model_repo_id = os.getenv(
        "GERMAN_MODEL_REPO_ID", "kikiri-tts/kikiri-german-martin"
    ).strip()
    model_filename = os.getenv(
        "GERMAN_MODEL_FILENAME", "kikiri_german_martin_ep10.pth"
    ).strip()
    model_subdir = os.getenv("GERMAN_MODEL_SUBDIR", "kikiri_german_martin").strip()

    voice_name = _strip_voice_extension(
        os.getenv("GERMAN_VOICE_FILENAME", "martin")
    ).strip()
    if not voice_name:
        voice_name = "martin"

    voice_repo_filename = os.getenv("GERMAN_VOICE_REPO_PATH", "voices/martin.pt").strip()
    config_repo_id = os.getenv("GERMAN_CONFIG_REPO_ID", "hexgrad/Kokoro-82M").strip()
    config_filename = os.getenv("GERMAN_CONFIG_FILENAME", "config.json").strip()

    model_ids_env = os.getenv("GERMAN_MODEL_IDS", "kikiri-german-martin")
    model_ids = tuple(
        item.strip() for item in model_ids_env.split(",") if item and item.strip()
    )

    default_profile_id = model_ids[0] if model_ids else "kikiri-german-martin"

    return ExternalProfile(
        profile_id=default_profile_id,
        model_repo_id=model_repo_id,
        model_filename=model_filename,
        model_subdir=model_subdir,
        model_ids=model_ids,
        voice_files={voice_name: voice_repo_filename},
        voice_repo_id=None,
        config_repo_id=config_repo_id,
        config_filename=config_filename,
        enabled=True,
    )


def load_external_profiles() -> list[ExternalProfile]:
    """Load external profiles from env (or legacy fallback)."""
    raw_profiles = os.getenv("EXTERNAL_MODEL_PROFILES", "").strip()
    profiles: list[ExternalProfile] = []

    if raw_profiles:
        try:
            parsed = json.loads(raw_profiles)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid EXTERNAL_MODEL_PROFILES JSON: {exc}"
            ) from exc

        if not isinstance(parsed, list):
            raise RuntimeError("EXTERNAL_MODEL_PROFILES must be a JSON list")

        for item in parsed:
            if not isinstance(item, dict):
                raise RuntimeError("Each EXTERNAL_MODEL_PROFILES item must be an object")

            profile = _profile_from_dict(item)
            if profile.enabled:
                profiles.append(profile)

        return profiles

    legacy_profile = _legacy_german_profile()
    if legacy_profile:
        profiles.append(legacy_profile)

    return profiles


def _select_external_profiles(
    profiles: list[ExternalProfile],
    requested_ids: set[str] | None,
) -> list[ExternalProfile]:
    if not requested_ids:
        return profiles

    selected: list[ExternalProfile] = []
    matched_ids: set[str] = set()

    for profile in profiles:
        candidate_ids = {_normalize_token(profile.profile_id)}
        candidate_ids.update(_normalize_token(item) for item in profile.model_ids)
        if requested_ids & candidate_ids:
            selected.append(profile)
            matched_ids.update(requested_ids & candidate_ids)

    missing = sorted(requested_ids - matched_ids)
    if missing:
        raise RuntimeError(
            "Requested external profile IDs not found: " + ", ".join(missing)
        )

    return selected


def _download_external_profile_assets(
    base_output: Path,
    profile: ExternalProfile,
    models_root_path: Path,
    voices_output_path: Path,
) -> None:
    model_dir = models_root_path / profile.model_subdir if profile.model_subdir else models_root_path
    model_path = model_dir / profile.model_filename
    config_path = model_dir / profile.config_filename

    if not _verify_non_empty_file(model_path):
        logger.info(
            f"Downloading model {profile.model_filename} from {profile.model_repo_id}"
        )
        _download_hf_file(profile.model_repo_id, profile.model_filename, model_path)

    if not _verify_json_file(config_path):
        base_config_path = base_output / BASE_CONFIG_FILE
        if _verify_json_file(base_config_path):
            logger.info(
                f"Copying base {BASE_CONFIG_FILE} for profile '{profile.profile_id}'"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base_config_path, config_path)
        else:
            logger.info(
                f"Downloading model config {profile.config_filename} from {profile.config_repo_id}"
            )
            _download_hf_file(profile.config_repo_id, profile.config_filename, config_path)

    voice_repo_id = profile.voice_repo_id or profile.model_repo_id
    for voice_name, repo_filename in profile.voice_files.items():
        voice_path = voices_output_path / f"{voice_name}.pt"
        if _verify_non_empty_file(voice_path):
            continue

        logger.info(f"Downloading voice {repo_filename} from {voice_repo_id}")
        _download_hf_file(voice_repo_id, repo_filename, voice_path)

    if not _verify_non_empty_file(model_path):
        raise RuntimeError(f"Failed to verify model file: {model_path.as_posix()}")
    if not _verify_json_file(config_path):
        raise RuntimeError(f"Failed to verify config file: {config_path.as_posix()}")

    for voice_name in profile.voice_files:
        voice_path = voices_output_path / f"{voice_name}.pt"
        if not _verify_non_empty_file(voice_path):
            raise RuntimeError(f"Failed to verify voice file: {voice_path.as_posix()}")


def download_external_profiles(
    base_output_dir: str,
    profiles: list[ExternalProfile],
    requested_ids: set[str] | None = None,
    models_root: str | None = None,
    voices_output: str | None = None,
) -> None:
    """Download optional external profile model and voice assets."""
    base_output = Path(base_output_dir).resolve()
    models_root_path = _resolve_models_root(base_output, models_root)
    voices_output_path = _resolve_voices_output(base_output, voices_output)

    selected_profiles = _select_external_profiles(profiles, requested_ids)
    if not selected_profiles:
        logger.warning("No enabled external profiles selected for download")
        return

    for profile in selected_profiles:
        logger.info(f"Preparing external profile assets: {profile.profile_id}")
        _download_external_profile_assets(
            base_output=base_output,
            profile=profile,
            models_root_path=models_root_path,
            voices_output_path=voices_output_path,
        )


def _parse_requested_profile_ids(raw_ids: str | None) -> set[str]:
    if not raw_ids:
        return set()
    return {
        _normalize_token(item)
        for item in raw_ids.split(",")
        if _normalize_token(item)
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download Kokoro model assets")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for base model files (example: api/src/models/v1_0)",
    )
    parser.add_argument(
        "--with-external-profiles",
        action="store_true",
        help="Also download external profile assets (from EXTERNAL_MODEL_PROFILES)",
    )
    parser.add_argument(
        "--external-profile-ids",
        default=None,
        help="Optional comma-separated external profile/model IDs to download",
    )
    parser.add_argument(
        "--with-german-martin",
        action="store_true",
        help="Backward-compatible shortcut for downloading the German Martin profile",
    )
    parser.add_argument(
        "--models-root",
        default=None,
        help=(
            "Optional models root directory used for external profiles "
            "(defaults to parent of --output)"
        ),
    )
    parser.add_argument(
        "--voices-output",
        default=None,
        help=(
            "Optional voices output directory used for external profiles "
            "(defaults to api/src/voices/v1_0 derived from --output)"
        ),
    )

    args = parser.parse_args()

    download_base_model(args.output)

    should_download_external = args.with_external_profiles or args.with_german_martin
    if not should_download_external:
        return

    profiles = load_external_profiles()

    if args.with_german_martin:
        has_martin_profile = any(
            "kikiri-german-martin"
            in ({_normalize_token(profile.profile_id)} | {_normalize_token(i) for i in profile.model_ids})
            for profile in profiles
        )
        if not has_martin_profile:
            legacy_profile = _legacy_german_profile()
            if legacy_profile:
                profiles.append(legacy_profile)

    requested_ids = _parse_requested_profile_ids(args.external_profile_ids)
    if args.with_german_martin:
        requested_ids.add("kikiri-german-martin")

    download_external_profiles(
        base_output_dir=args.output,
        profiles=profiles,
        requested_ids=requested_ids if requested_ids else None,
        models_root=args.models_root,
        voices_output=args.voices_output,
    )


if __name__ == "__main__":
    main()
