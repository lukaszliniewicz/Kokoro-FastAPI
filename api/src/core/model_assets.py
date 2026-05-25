"""Model/voice asset resolution and lazy download helpers."""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from loguru import logger

from .config import ExternalModelProfile, settings
from .model_config import model_config

# Keep this in sync with kokoro/misaki language aliases.
LANG_CODE_ALIASES = {
    "en-us": "a",
    "en-gb": "b",
    "de": "d",
    "es": "e",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
    "ja": "j",
    "zh": "z",
}

_asset_lock: asyncio.Lock | None = None
_asset_lock_loop: asyncio.AbstractEventLoop | None = None


@dataclass(frozen=True)
class _ProfileView:
    profile: ExternalModelProfile
    profile_id: str
    model_relative_path: str
    model_ids: frozenset[str]
    lang_codes: frozenset[str]
    default_lang_code: str | None
    voice_lookup: frozenset[str]
    voice_canonical_map: dict[str, str]
    voice_files: dict[str, str]


def _normalize_token(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower()


def _strip_voice_extension(voice_name: str) -> str:
    value = voice_name.strip()
    if value.lower().endswith(".pt"):
        return value[:-3]
    return value


def normalize_lang_code(lang_code: str | None) -> str | None:
    """Normalize language code aliases to kokoro short codes."""
    normalized = _normalize_token(lang_code)
    if not normalized:
        return None
    return LANG_CODE_ALIASES.get(normalized, normalized)


def parse_voice_names(voice_expression: str) -> list[str]:
    """Extract base voice names from a weighted voice expression."""
    if not voice_expression:
        return []

    names: list[str] = []
    parts = re.split(r"[-+]", voice_expression)
    for part in parts:
        item = part.strip()
        if not item:
            continue
        name = item.split("(", 1)[0].strip().lower()
        if name:
            names.append(name)
    return names


def _legacy_german_profile() -> ExternalModelProfile | None:
    if not settings.enable_german_martin_support:
        return None

    voice_name = _strip_voice_extension(settings.german_voice_name).strip().lower()
    if not voice_name:
        return None

    default_profile_id = next(
        (item.strip() for item in settings.german_model_ids if item and item.strip()),
        settings.german_model_subdir,
    )

    return ExternalModelProfile(
        enabled=True,
        profile_id=default_profile_id,
        model_repo_id=settings.german_model_repo_id,
        model_filename=settings.german_model_filename,
        model_subdir=settings.german_model_subdir,
        model_ids=list(settings.german_model_ids),
        lang_codes=["de", "d"],
        voice_files={voice_name: settings.german_voice_filename_in_repo},
        voice_names=[voice_name],
        voice_aliases={voice_name: list(settings.german_voice_aliases)},
        config_repo_id=settings.german_config_repo_id,
        config_filename=settings.german_config_filename,
    )


def _is_same_profile(a: ExternalModelProfile, b: ExternalModelProfile) -> bool:
    a_profile_id = _normalize_token(a.profile_id)
    b_profile_id = _normalize_token(b.profile_id)
    if a_profile_id and b_profile_id and a_profile_id == b_profile_id:
        return True

    a_path = _normalize_token(a.model_subdir) + "/" + _normalize_token(a.model_filename)
    b_path = _normalize_token(b.model_subdir) + "/" + _normalize_token(b.model_filename)
    if a_path == b_path:
        return True

    a_model_ids = {_normalize_token(item) for item in a.model_ids if _normalize_token(item)}
    b_model_ids = {_normalize_token(item) for item in b.model_ids if _normalize_token(item)}
    return bool(a_model_ids & b_model_ids)


def get_external_model_profiles() -> list[ExternalModelProfile]:
    """Return active external profile configs (explicit + legacy fallback)."""
    profiles: list[ExternalModelProfile] = []

    for candidate in settings.external_model_profiles:
        try:
            profile = (
                candidate
                if isinstance(candidate, ExternalModelProfile)
                else ExternalModelProfile.model_validate(candidate)
            )
        except Exception as exc:
            logger.warning(f"Skipping invalid external model profile entry: {exc}")
            continue

        if profile.enabled:
            profiles.append(profile)

    legacy_profile = _legacy_german_profile()
    if legacy_profile and not any(
        _is_same_profile(existing, legacy_profile) for existing in profiles
    ):
        profiles.append(legacy_profile)

    return profiles


def _normalize_voice_files(profile: ExternalModelProfile) -> dict[str, str]:
    voice_files: dict[str, str] = {}
    for local_voice_name, repo_filename in profile.voice_files.items():
        canonical_name = _normalize_token(_strip_voice_extension(local_voice_name))
        repo_path = repo_filename.strip()
        if canonical_name and repo_path:
            voice_files[canonical_name] = repo_path
    return voice_files


def _join_model_relative_path(subdir: str, filename: str) -> str:
    filename_value = filename.strip()
    subdir_value = subdir.strip()
    if not subdir_value:
        return filename_value
    return str(Path(subdir_value) / filename_value)


def _build_profile_view(profile: ExternalModelProfile) -> _ProfileView:
    profile_id = _normalize_token(profile.profile_id)
    model_ids = {
        _normalize_token(item) for item in profile.model_ids if _normalize_token(item)
    }
    if profile_id:
        model_ids.add(profile_id)

    normalized_lang_codes: list[str] = []
    for value in profile.lang_codes:
        normalized = normalize_lang_code(value)
        if normalized:
            normalized_lang_codes.append(normalized)

    default_lang_code = normalized_lang_codes[0] if normalized_lang_codes else None

    voice_files = _normalize_voice_files(profile)
    voice_name_set = {
        _normalize_token(_strip_voice_extension(item))
        for item in profile.voice_names
        if _normalize_token(_strip_voice_extension(item))
    }

    voice_canonical_map: dict[str, str] = {}
    for canonical_name in voice_files:
        voice_canonical_map[canonical_name] = canonical_name
    for voice_name in voice_name_set:
        voice_canonical_map.setdefault(voice_name, voice_name)

    for base_name, aliases in profile.voice_aliases.items():
        canonical_name = _normalize_token(_strip_voice_extension(base_name))
        if not canonical_name:
            continue

        voice_canonical_map.setdefault(canonical_name, canonical_name)
        for alias in aliases:
            alias_name = _normalize_token(_strip_voice_extension(alias))
            if alias_name:
                voice_canonical_map[alias_name] = canonical_name

    voice_lookup = set(voice_canonical_map.keys())
    voice_lookup.update(voice_name_set)
    voice_lookup.update(voice_files.keys())

    return _ProfileView(
        profile=profile,
        profile_id=profile_id,
        model_relative_path=_join_model_relative_path(
            profile.model_subdir, profile.model_filename
        ),
        model_ids=frozenset(model_ids),
        lang_codes=frozenset(normalized_lang_codes),
        default_lang_code=default_lang_code,
        voice_lookup=frozenset(voice_lookup),
        voice_canonical_map=voice_canonical_map,
        voice_files=voice_files,
    )


def _profile_views() -> list[_ProfileView]:
    return [_build_profile_view(profile) for profile in get_external_model_profiles()]


def get_external_profile_model_mappings() -> dict[str, str]:
    """Return OpenAI model-id mappings derived from external profiles."""
    mappings: dict[str, str] = {}
    for view in _profile_views():
        internal_name = Path(view.profile.model_filename).stem
        for model_id in view.model_ids:
            if model_id:
                mappings[model_id] = internal_name
    return mappings


def list_external_profile_voices() -> list[str]:
    """Return configured voice names from active external profiles."""
    voices: set[str] = set()
    for view in _profile_views():
        voices.update(view.voice_files.keys())
        for item in view.profile.voice_names:
            name = _normalize_token(_strip_voice_extension(item))
            if name:
                voices.add(name)
    return sorted(voices)


def canonicalize_voice_name(voice_name: str) -> str:
    """Resolve configured voice aliases to canonical voice names."""
    normalized = _normalize_token(_strip_voice_extension(voice_name or ""))
    if not normalized:
        return voice_name

    for view in _profile_views():
        canonical = view.voice_canonical_map.get(normalized)
        if canonical:
            return canonical

    return normalized


def canonicalize_voice_expression(voice_expression: str) -> str:
    """Resolve aliases in a weighted voice expression."""
    if not voice_expression:
        return voice_expression

    parts = re.split(r"([-+])", voice_expression)
    for index in range(0, len(parts), 2):
        token = parts[index].strip()
        if not token:
            continue

        base_name = token
        suffix = ""
        if "(" in token:
            base_name, remainder = token.split("(", 1)
            suffix = "(" + remainder

        parts[index] = f"{canonicalize_voice_name(base_name)}{suffix}"

    return "".join(parts)


def _select_profile_by_voice(voice_expression: str) -> _ProfileView | None:
    requested_names = set(parse_voice_names(canonicalize_voice_expression(voice_expression)))
    if not requested_names:
        return None

    for view in _profile_views():
        if requested_names & view.voice_lookup:
            return view
    return None


def _select_profile_for_request(
    voice_expression: str,
    resolved_lang_code: str | None,
    requested_model: str | None,
) -> _ProfileView | None:
    normalized_model = _normalize_token(requested_model)
    normalized_lang = normalize_lang_code(resolved_lang_code)
    requested_voice_names = set(
        parse_voice_names(canonicalize_voice_expression(voice_expression))
    )

    views = _profile_views()

    if normalized_model:
        for view in views:
            if normalized_model in view.model_ids:
                return view

    if normalized_lang:
        for view in views:
            if normalized_lang in view.lang_codes:
                return view

    if requested_voice_names:
        for view in views:
            if requested_voice_names & view.voice_lookup:
                return view

    return None


def resolve_pipeline_lang_code(
    voice_expression: str,
    requested_lang_code: str | None,
) -> str:
    """Resolve the effective pipeline language code for a request."""
    normalized = normalize_lang_code(requested_lang_code)
    if normalized:
        return normalized

    voice_profile = _select_profile_by_voice(voice_expression)
    if voice_profile and voice_profile.default_lang_code:
        return voice_profile.default_lang_code

    if settings.default_voice_code:
        default_code = normalize_lang_code(settings.default_voice_code)
        if default_code:
            return default_code

    parsed_names = parse_voice_names(canonicalize_voice_expression(voice_expression))
    if parsed_names and parsed_names[0]:
        return normalize_lang_code(parsed_names[0][0]) or parsed_names[0][0]

    return "a"


def resolve_model_file_for_request(
    voice_expression: str,
    resolved_lang_code: str | None,
    requested_model: str | None,
) -> str:
    """Resolve model filename (relative to MODEL_DIR) for the request."""
    profile = _select_profile_for_request(
        voice_expression=voice_expression,
        resolved_lang_code=resolved_lang_code,
        requested_model=requested_model,
    )
    if profile:
        return profile.model_relative_path

    return model_config.pytorch_kokoro_v1_file


def _resolve_api_subdir(path_setting: str) -> Path:
    """Resolve an API-relative directory setting to an absolute path."""
    api_dir = Path(__file__).resolve().parents[2]
    configured = Path(path_setting)
    resolved = configured if configured.is_absolute() else api_dir / configured
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _default_config_candidate(model_root: Path, config_filename: str) -> Path:
    model_file = Path(model_config.pytorch_kokoro_v1_file)
    model_path = model_file if model_file.is_absolute() else model_root / model_file
    return model_path.parent / config_filename


def _profile_asset_paths(view: _ProfileView) -> tuple[Path, Path, dict[str, Path]]:
    model_dir = _resolve_api_subdir(settings.model_dir)
    voice_dir = _resolve_api_subdir(settings.voices_dir)

    model_path = model_dir / view.model_relative_path
    config_path = model_path.parent / view.profile.config_filename
    voice_paths = {
        voice_name: voice_dir / f"{voice_name}.pt" for voice_name in view.voice_files
    }
    return model_path, config_path, voice_paths


def _download_to_path(repo_id: str, filename: str, target_path: Path) -> None:
    source_path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _requested_profile_voices(view: _ProfileView, voice_expression: str) -> set[str]:
    requested = set(parse_voice_names(canonicalize_voice_expression(voice_expression)))
    resolved: set[str] = set()
    for name in requested:
        canonical = view.voice_canonical_map.get(name, name)
        if canonical in view.voice_files:
            resolved.add(canonical)
    return resolved


def _ensure_profile_assets_sync(
    view: _ProfileView,
    required_voice_names: set[str],
) -> None:
    model_path, config_path, voice_paths = _profile_asset_paths(view)

    if not model_path.exists():
        logger.info(
            f"Downloading model '{view.profile.model_filename}' from '{view.profile.model_repo_id}'"
        )
        _download_to_path(
            view.profile.model_repo_id,
            view.profile.model_filename,
            model_path,
        )

    if not config_path.exists():
        fallback_config = _default_config_candidate(
            _resolve_api_subdir(settings.model_dir),
            view.profile.config_filename,
        )
        if fallback_config.exists() and fallback_config.resolve() != config_path.resolve():
            logger.info(
                f"Copying model config from existing file: {fallback_config.as_posix()}"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback_config, config_path)
        else:
            logger.info(
                f"Downloading model config '{view.profile.config_filename}' from '{view.profile.config_repo_id}'"
            )
            _download_to_path(
                view.profile.config_repo_id,
                view.profile.config_filename,
                config_path,
            )

    voice_repo_id = view.profile.voice_repo_id or view.profile.model_repo_id
    for voice_name in required_voice_names:
        voice_path = voice_paths.get(voice_name)
        voice_repo_filename = view.voice_files.get(voice_name)
        if not voice_path or not voice_repo_filename:
            continue

        if not voice_path.exists():
            logger.info(
                f"Downloading voice '{voice_repo_filename}' from '{voice_repo_id}'"
            )
            _download_to_path(voice_repo_id, voice_repo_filename, voice_path)


def _verify_profile_assets_present(
    view: _ProfileView,
    required_voice_names: set[str],
) -> None:
    model_path, config_path, voice_paths = _profile_asset_paths(view)
    missing: list[str] = []

    if not model_path.exists():
        missing.append(model_path.as_posix())
    if not config_path.exists():
        missing.append(config_path.as_posix())

    for voice_name in required_voice_names:
        voice_path = voice_paths.get(voice_name)
        if voice_path and not voice_path.exists():
            missing.append(voice_path.as_posix())

    if missing:
        missing_paths = ", ".join(missing)
        raise FileNotFoundError(
            "External model assets are missing and auto-download is disabled. "
            f"Profile: {view.profile_id or view.model_relative_path}. "
            f"Missing: {missing_paths}"
        )


def _get_asset_lock() -> asyncio.Lock:
    global _asset_lock
    global _asset_lock_loop

    loop = asyncio.get_running_loop()
    if _asset_lock is None or _asset_lock_loop is not loop:
        _asset_lock = asyncio.Lock()
        _asset_lock_loop = loop
    return _asset_lock


async def ensure_assets_for_request(
    voice_expression: str,
    resolved_lang_code: str | None,
    requested_model: str | None,
) -> None:
    """Ensure dynamic assets required by a request exist locally."""
    profile = _select_profile_for_request(
        voice_expression=voice_expression,
        resolved_lang_code=resolved_lang_code,
        requested_model=requested_model,
    )
    if not profile:
        return

    required_voices = _requested_profile_voices(profile, voice_expression)

    if settings.auto_download_model_assets:
        lock = _get_asset_lock()
        async with lock:
            await asyncio.to_thread(
                _ensure_profile_assets_sync,
                profile,
                required_voices,
            )
    else:
        _verify_profile_assets_present(profile, required_voices)
