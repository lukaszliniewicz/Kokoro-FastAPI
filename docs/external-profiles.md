# External Model Profiles

This fork supports routing requests to additional fine-tuned models through
`EXTERNAL_MODEL_PROFILES`.

Each profile defines:
- where model/voice assets are downloaded from,
- which OpenAI-style model IDs trigger it,
- optional language hints,
- local voice names and aliases.

The same API instance can serve base Kokoro plus multiple fine-tuned profiles.

## Routing Priority

For each request, profile selection is evaluated in this order:

1. `model` ID match (exact configured profile model ID)
2. `lang_code` match (profile `lang_codes`, normalized)
3. `voice` match (configured `voice_files` / aliases)

If nothing matches, the request uses the base Kokoro model.

## Profile Schema

Set `EXTERNAL_MODEL_PROFILES` to a JSON list.

Required fields:
- `profile_id` (string)
- `model_repo_id` (string, Hugging Face repo)
- `model_filename` (string, model filename in repo)
- `model_subdir` (string, local model subdirectory under `MODEL_DIR`)

Common optional fields:
- `enabled` (bool, default `true`)
- `model_ids` (list of model IDs accepted by `/v1/audio/speech`)
- `lang_codes` (list, for lang-based routing)
- `voice_files` (object: local voice name -> repo file path)
- `voice_names` (list of local voice names to advertise)
- `voice_aliases` (object: canonical voice name -> list of aliases)
- `voice_repo_id` (string, if voices are in a different repo)
- `config_repo_id` (string, default `hexgrad/Kokoro-82M`)
- `config_filename` (string, default `config.json`)

## Minimal Example

```bash
EXTERNAL_MODEL_PROFILES='[
  {
    "profile_id": "my-finetune",
    "model_repo_id": "org/my-finetune-repo",
    "model_filename": "my_finetune.pth",
    "model_subdir": "my_finetune",
    "model_ids": ["my-finetune"],
    "lang_codes": ["en-us", "a"],
    "voice_files": {
      "my_voice": "voices/my_voice.pt"
    },
    "voice_aliases": {
      "my_voice": ["my_voice", "my_voice_v1"]
    }
  }
]'
```

## Multi-Voice Example

```json
[
  {
    "profile_id": "my-multi-voice-finetune",
    "model_repo_id": "org/my-finetune-repo",
    "model_filename": "my_finetune.pth",
    "model_subdir": "my_finetune",
    "model_ids": ["my-finetune", "my-finetune-hd"],
    "lang_codes": ["de", "d"],
    "voice_files": {
      "voice_a": "voices/voice_a.pt",
      "voice_b": "voices/voice_b.pt"
    },
    "voice_aliases": {
      "voice_a": ["a", "voice_a"],
      "voice_b": ["b", "voice_b"]
    }
  }
]
```

## Separate Voice Repo Example

```json
[
  {
    "profile_id": "my-model-with-external-voices",
    "model_repo_id": "org/model-repo",
    "model_filename": "model.pth",
    "model_subdir": "model",
    "model_ids": ["my-model"],
    "voice_repo_id": "org/voice-repo",
    "voice_files": {
      "my_voice": "packs/my_voice.pt"
    }
  }
]
```

## Prefetch vs Lazy Download

Runtime lazy download:
- `AUTO_DOWNLOAD_MODEL_ASSETS=true` (default)

Startup prefetch flags:
- `DOWNLOAD_EXTERNAL_MODELS=true` -> download all enabled profiles
- `DOWNLOAD_EXTERNAL_PROFILE_IDS=profile-a,profile-b` -> download subset only
- `DOWNLOAD_GERMAN_MARTIN=true` -> legacy shortcut for Martin profile

These flags are supported in startup scripts and Docker entrypoint flow.

## API Behavior

- `/v1/models` is auto-extended with configured profile model IDs.
- `/v1/audio/voices` includes configured profile voice names when
  auto-download is enabled.
- Voice aliases are resolved to canonical voice names before generation.

## Quick Request Example

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-finetune",
    "voice": "my_voice_v1",
    "input": "Hello from my fine-tuned profile.",
    "response_format": "wav",
    "stream": false
  }' --output out.wav
```

## Backward Compatibility

Legacy Martin settings are still supported (`GERMAN_*` +
`ENABLE_GERMAN_MARTIN_SUPPORT`).

If `EXTERNAL_MODEL_PROFILES` is not set, the legacy Martin config is mapped to
one implicit external profile.

## Troubleshooting

- If startup fails with profile parsing errors, validate JSON formatting in
  `EXTERNAL_MODEL_PROFILES`.
- If model ID is rejected, confirm it appears in `/v1/models`.
- If voice is rejected, confirm it appears in `/v1/audio/voices` and alias maps
  to a configured canonical voice.
- If using profile subsets, ensure `DOWNLOAD_EXTERNAL_PROFILE_IDS` values match
  `profile_id` or one of that profile's `model_ids`.
