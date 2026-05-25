"""Kokoro V1 model management."""

import asyncio
from typing import Optional

from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_assets import (
    ensure_assets_for_request,
    resolve_model_file_for_request,
    resolve_pipeline_lang_code,
)
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend
from .kokoro_v1 import KokoroV1


class ModelManager:
    """Manages Kokoro V1 model loading and inference."""

    # Singleton instance
    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize manager.

        Args:
            config: Optional model configuration override
        """
        self._config = config or model_config
        self._backend: Optional[KokoroV1] = None  # Explicitly type as KokoroV1
        self._device: Optional[str] = None
        self._loaded_model_path: Optional[str] = None
        self._model_lock: Optional[asyncio.Lock] = None
        self._model_lock_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_model_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._model_lock is None or self._model_lock_loop is not loop:
            self._model_lock = asyncio.Lock()
            self._model_lock_loop = loop
        return self._model_lock

    def _determine_device(self) -> str:
        """Determine device based on settings."""
        return "cuda" if settings.use_gpu else "cpu"

    async def initialize(self) -> None:
        """Initialize Kokoro V1 backend."""
        try:
            self._device = self._determine_device()
            logger.info(f"Initializing Kokoro V1 on {self._device}")
            self._backend = KokoroV1()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kokoro V1: {e}")

    async def initialize_with_warmup(self, voice_manager) -> tuple[str, str, int]:
        """Initialize and warm up model.

        Args:
            voice_manager: Voice manager instance for warmup

        Returns:
            Tuple of (device, backend type, voice count)

        Raises:
            RuntimeError: If initialization fails
        """
        import time

        start = time.perf_counter()

        try:
            # Initialize backend
            await self.initialize()

            # Resolve and load startup model based on configured default voice/lang
            startup_lang_code = resolve_pipeline_lang_code(
                settings.default_voice, settings.default_voice_code
            )
            model_path = resolve_model_file_for_request(
                settings.default_voice,
                startup_lang_code,
                None,
            )
            await ensure_assets_for_request(
                settings.default_voice,
                startup_lang_code,
                None,
            )
            await self.ensure_model_loaded(model_path)

            # Use paths module to get voice path
            try:
                voices = await paths.list_voices()
                voice_path = await paths.get_voice_path(settings.default_voice)

                # Warm up with short text
                warmup_text = "Warmup text for initialization."
                # Use default voice name for warmup
                voice_name = settings.default_voice
                logger.debug(f"Using default voice '{voice_name}' for warmup")
                async for _ in self.generate(
                    warmup_text,
                    (voice_name, voice_path),
                    lang_code=startup_lang_code,
                ):
                    pass
            except Exception as e:
                raise RuntimeError(f"Failed to get default voice: {e}")

            ms = int((time.perf_counter() - start) * 1000)
            logger.info(f"Warmup completed in {ms}ms")

            return self._device, "kokoro_v1", len(voices)
        except FileNotFoundError as e:
            logger.error("""
Model files not found! You need to download the Kokoro V1 model:

1. Download model using the script:
   python docker/scripts/download_model.py --output api/src/models/v1_0

2. Or set environment variable in docker-compose:
   DOWNLOAD_MODEL=true
""")
            exit(0)
        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")

    def get_backend(self) -> BaseModelBackend:
        """Get initialized backend.

        Returns:
            Initialized backend instance

        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")
        return self._backend

    async def load_model(self, path: str) -> None:
        """Load model using initialized backend.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If loading fails
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            await self._backend.load_model(path)
            self._loaded_model_path = path
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    async def ensure_model_loaded(self, path: str) -> None:
        """Ensure the requested model is loaded and active."""
        async with self._get_model_lock():
            if not self._backend:
                await self.initialize()

            if (
                self._backend
                and self._backend.is_loaded
                and self._loaded_model_path == path
            ):
                return

            if self._backend and self._backend.is_loaded:
                logger.info(f"Switching active model from '{self._loaded_model_path}' to '{path}'")
                self._backend.unload()

            await self.load_model(path)

    async def generate(self, *args, **kwargs):
        """Generate audio using initialized backend.

        Raises:
            RuntimeError: If generation fails
        """
        async with self._get_model_lock():
            if not self._backend:
                raise RuntimeError("Backend not initialized")

            try:
                async for chunk in self._backend.generate(*args, **kwargs):
                    if settings.default_volume_multiplier != 1.0:
                        chunk.audio *= settings.default_volume_multiplier
                    yield chunk
            except Exception as e:
                raise RuntimeError(f"Generation failed: {e}")

    def unload_all(self) -> None:
        """Unload model and free resources."""
        if self._backend:
            self._backend.unload()
            self._backend = None

    @property
    def current_backend(self) -> str:
        """Get current backend type."""
        return "kokoro_v1"


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Get model manager instance.

    Args:
        config: Optional configuration override

    Returns:
        ModelManager instance
    """
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager(config)
    return ModelManager._instance
