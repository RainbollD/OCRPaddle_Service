import asyncio
import logging
import time

import docker
import httpx
from docker.errors import NotFound

from app.config import settings
from app.registry import MODELS, ModelSpec

logger = logging.getLogger(__name__)


class BackendError(Exception):
    """Raised when a backend container cannot be started or fails to become ready."""


class BackendManager:
    """Starts/stops one model backend at a time and tracks the active model.

    The server GPU only holds a single model, so switching models means stopping
    the currently running backend before starting the next one. A lock
    serializes swaps so concurrent requests cannot fight over the GPU.
    """

    def __init__(self) -> None:
        self._client = docker.from_env()
        self._lock = asyncio.Lock()
        self._active: str | None = None

    @property
    def active(self) -> str | None:
        """Name of the model whose backend is currently loaded, if any."""
        return self._active

    async def ensure_only(self, spec: ModelSpec) -> None:
        """Ensure ``spec``'s backend is the only one running, and is ready.

        Stops every other known backend (freeing VRAM), starts the requested
        one if needed, and waits until its ``/health`` endpoint responds.

        Args:
            spec: The model whose backend must be running.

        Raises:
            BackendError: If the container is missing or never becomes ready.
        """
        async with self._lock:
            if self._active == spec.name and await self._is_ready(spec):
                return

            await asyncio.to_thread(self._stop_others, spec.name)
            self._active = None

            await asyncio.to_thread(self._start, spec)
            await self._wait_ready(spec)
            self._active = spec.name
            logger.info("Backend '%s' is active and ready.", spec.name)

    async def stop(self, name: str) -> None:
        """Stop a single backend by model name (best effort)."""
        async with self._lock:
            await asyncio.to_thread(self._stop_one, name)
            if self._active == name:
                self._active = None

    def _stop_others(self, keep: str) -> None:
        for name, spec in MODELS.items():
            if name != keep:
                self._stop_one(name)

    def _stop_one(self, name: str) -> None:
        spec = MODELS[name]
        try:
            container = self._client.containers.get(spec.container)
        except NotFound:
            return
        if container.status == "running":
            logger.info("Stopping backend '%s' (%s).", name, spec.container)
            container.stop(timeout=30)

    def _start(self, spec: ModelSpec) -> None:
        try:
            container = self._client.containers.get(spec.container)
        except NotFound as exc:
            raise BackendError(
                f"Backend container '{spec.container}' not found. Create it first "
                f"with: docker compose --profile backend up --no-start {spec.service}"
            ) from exc
        container.reload()
        if container.status != "running":
            logger.info("Starting backend '%s' (%s).", spec.name, spec.container)
            container.start()

    async def _is_ready(self, spec: ModelSpec) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{spec.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def _wait_ready(self, spec: ModelSpec) -> None:
        deadline = time.monotonic() + settings.backend_ready_timeout_s
        while time.monotonic() < deadline:
            if await self._is_ready(spec):
                return
            await asyncio.sleep(settings.backend_poll_interval_s)
        raise BackendError(
            f"Backend '{spec.name}' did not become ready within "
            f"{settings.backend_ready_timeout_s}s."
        )
