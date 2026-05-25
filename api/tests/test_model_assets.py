from __future__ import annotations

from pathlib import Path

import pytest

from api.src.core import model_assets
from api.src.core.config import ExternalModelProfile


def _profile(
    profile_id: str = "kikiri-german-martin",
    model_filename: str = "kikiri_german_martin_ep10.pth",
    model_subdir: str = "kikiri_german_martin",
    model_ids: list[str] | None = None,
    lang_codes: list[str] | None = None,
    voice_files: dict[str, str] | None = None,
    voice_aliases: dict[str, list[str]] | None = None,
) -> ExternalModelProfile:
    return ExternalModelProfile(
        profile_id=profile_id,
        model_repo_id="kikiri-tts/kikiri-german-martin",
        model_filename=model_filename,
        model_subdir=model_subdir,
        model_ids=model_ids or ["kikiri-german-martin"],
        lang_codes=lang_codes or ["de", "d"],
        voice_files=voice_files or {"martin": "voices/martin.pt"},
        voice_aliases=voice_aliases or {"martin": ["martin"]},
        config_repo_id="hexgrad/Kokoro-82M",
        config_filename="config.json",
    )


def test_parse_voice_names_with_weights():
    voices = model_assets.parse_voice_names("af_bella(2)+martin(1)-bf_emma")
    assert voices == ["af_bella", "martin", "bf_emma"]


def test_resolve_pipeline_lang_code_from_voice(monkeypatch):
    monkeypatch.setattr(model_assets.settings, "enable_german_martin_support", False)
    monkeypatch.setattr(
        model_assets.settings,
        "external_model_profiles",
        [_profile()],
    )

    assert model_assets.resolve_pipeline_lang_code("martin", None) == "d"


def test_canonicalize_voice_expression_alias(monkeypatch):
    monkeypatch.setattr(model_assets.settings, "enable_german_martin_support", False)
    monkeypatch.setattr(
        model_assets.settings,
        "external_model_profiles",
        [_profile(voice_aliases={"martin": ["de_martin", "martin"]})],
    )

    assert model_assets.canonicalize_voice_name("de_martin") == "martin"
    assert (
        model_assets.canonicalize_voice_expression("de_martin(2)+af_heart")
        == "martin(2)+af_heart"
    )


def test_resolve_pipeline_lang_code_alias():
    assert model_assets.normalize_lang_code("de") == "d"
    assert model_assets.normalize_lang_code("en-us") == "a"


def test_resolve_model_file_for_german_model(monkeypatch):
    monkeypatch.setattr(model_assets.settings, "enable_german_martin_support", False)
    monkeypatch.setattr(
        model_assets.settings,
        "external_model_profiles",
        [
            _profile(
                model_subdir="kikiri",
                model_filename="martin.pth",
                model_ids=["kikiri-german-martin"],
            )
        ],
    )

    path = model_assets.resolve_model_file_for_request(
        voice_expression="af_heart",
        resolved_lang_code="a",
        requested_model="kikiri-german-martin",
    )
    assert Path(path).as_posix() == "kikiri/martin.pth"


@pytest.mark.asyncio
async def test_ensure_assets_for_request_missing_when_download_disabled(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(model_assets.settings, "enable_german_martin_support", False)
    monkeypatch.setattr(model_assets.settings, "auto_download_model_assets", False)
    monkeypatch.setattr(
        model_assets.settings,
        "external_model_profiles",
        [_profile()],
    )

    model_path = tmp_path / "missing" / "model.pth"
    config_path = tmp_path / "missing" / "config.json"
    voice_path = tmp_path / "missing" / "martin.pt"
    monkeypatch.setattr(
        model_assets,
        "_profile_asset_paths",
        lambda _view: (model_path, config_path, {"martin": voice_path}),
    )

    with pytest.raises(FileNotFoundError, match="External model assets are missing"):
        await model_assets.ensure_assets_for_request("martin", "d", None)


@pytest.mark.asyncio
async def test_ensure_assets_for_request_triggers_download(monkeypatch):
    monkeypatch.setattr(model_assets.settings, "enable_german_martin_support", False)
    monkeypatch.setattr(model_assets.settings, "auto_download_model_assets", True)
    monkeypatch.setattr(
        model_assets.settings,
        "external_model_profiles",
        [_profile()],
    )

    calls = {"count": 0, "voices": set()}

    def fake_download(_view, required_voices):
        calls["count"] += 1
        calls["voices"] = set(required_voices)

    monkeypatch.setattr(model_assets, "_ensure_profile_assets_sync", fake_download)

    await model_assets.ensure_assets_for_request("martin", "d", None)
    assert calls["count"] == 1
    assert calls["voices"] == {"martin"}
