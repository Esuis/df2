"""ELLM ApiKey Manager - Dynamic API key refresh for BOCOM ELLM gateway.

This module provides a thread-safe singleton that automatically refreshes
the API key for the BOCOM ELLM (Enterprise Large Language Model) gateway.

Key features:
  - Background thread refreshes the API key every ``refresh_interval`` seconds
  - ``get_api_key()`` returns the latest valid key, auto-refreshing if near expiry
  - Per ``scene_code`` singleton instances to avoid duplicate requests
  - Graceful degradation: keeps the old key if refresh fails
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default refresh configuration
DEFAULT_REFRESH_INTERVAL = 1800  # 25 minutes in seconds
DEFAULT_REFRESH_AHEAD = 300  # Refresh 2 minutes before expiry
DEFAULT_REQUEST_TIMEOUT = 30  # 30 seconds timeout for key request


class EllmApiKeyManager:
    """Manages ELLM API keys with automatic background refresh.

    A per-scene_code singleton that periodically fetches a new API key from
    the ELLM gateway and stores it in memory. Callers use ``get_api_key()``
    to obtain the current valid key — the key is refreshed transparently
    without requiring a process restart.

    Usage::

        manager = EllmApiKeyManager.get_instance(
            api_key_url="http://eaip-ellm-1.bocomm.com/ELLM.ELLM-OMSERVICE.V-1.0/createSceneApiKey.do",
            scene_code="P2024146",
        )
        manager.start()
        key = manager.get_api_key()
    """

    # Class-level registry: scene_code -> EllmApiKeyManager
    _instances: dict[str, EllmApiKeyManager] = {}
    _class_lock = threading.Lock()

    def __init__(
        self,
        api_key_url: str,
        scene_code: str,
        refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
        refresh_ahead: int = DEFAULT_REFRESH_AHEAD,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self._api_key_url = api_key_url
        self._scene_code = scene_code
        self._refresh_interval = refresh_interval
        self._refresh_ahead = refresh_ahead
        self._request_timeout = request_timeout

        # Internal state
        self._current_key: str = ""
        self._key_obtained_at: float = 0.0  # timestamp when key was obtained
        self._key_ttl_ms: int = 0  # TTL in milliseconds from the response
        self._lock = threading.Lock()
        self._refresh_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

        # HTTP client (lazy init)
        self._http_client: httpx.Client | None = None

    @classmethod
    def get_instance(
        cls,
        api_key_url: str,
        scene_code: str,
        refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
        refresh_ahead: int = DEFAULT_REFRESH_AHEAD,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ) -> EllmApiKeyManager:
        """Get or create the singleton manager for a given scene_code.

        If a manager for the given scene_code already exists, the existing
        instance is returned (configuration parameters are ignored).
        """
        with cls._class_lock:
            if scene_code not in cls._instances:
                instance = cls(
                    api_key_url=api_key_url,
                    scene_code=scene_code,
                    refresh_interval=refresh_interval,
                    refresh_ahead=refresh_ahead,
                    request_timeout=request_timeout,
                )
                cls._instances[scene_code] = instance
            return cls._instances[scene_code]

    @classmethod
    def get_instance_by_scene_code(cls, scene_code: str) -> EllmApiKeyManager | None:
        """Get the existing manager for a scene_code, or None if not created."""
        with cls._class_lock:
            return cls._instances.get(scene_code)

    def start(self) -> None:
        """Start the background refresh thread.

        Safe to call multiple times — only starts once.
        Also performs an initial synchronous key fetch if no key is available.
        """
        with self._lock:
            if self._started:
                return
            self._started = True

        # Initial synchronous key fetch
        try:
            self.refresh_key()
            logger.info(
                "ELLM ApiKeyManager started for scene_code=%s, initial key obtained",
                self._scene_code,
            )
        except Exception:
            logger.warning(
                "ELLM ApiKeyManager: initial key fetch failed for scene_code=%s, "
                "will retry in background thread",
                self._scene_code,
            )

        # Start background refresh thread
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            name=f"ellm-apikey-refresh-{self._scene_code}",
            daemon=True,
        )
        self._refresh_thread.start()

        # Register cleanup on process exit
        atexit.register(self.stop)

    def stop(self) -> None:
        """Stop the background refresh thread."""
        self._stop_event.set()
        with self._lock:
            self._started = False
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5)
            self._refresh_thread = None
        # Close HTTP client
        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:
                pass
            self._http_client = None

    def get_api_key(self) -> str:
        """Get the current valid API key.

        If the key is near expiry (within ``refresh_ahead`` seconds), a
        synchronous refresh is attempted before returning. If no key is
        available at all, a synchronous refresh is forced.

        Returns:
            The current valid API key string.

        Raises:
            RuntimeError: If no key is available and refresh fails.
        """
        with self._lock:
            if self._current_key and not self._is_near_expiry():
                return self._current_key

        # Key is near expiry or missing — try synchronous refresh
        try:
            return self.refresh_key()
        except Exception as e:
            # If we still have a key (even near-expiry), return it
            with self._lock:
                if self._current_key:
                    logger.warning(
                        "ELLM ApiKeyManager: key refresh failed, using existing key "
                        "(scene_code=%s, error=%s)",
                        self._scene_code,
                        e,
                    )
                    return self._current_key
            # No key at all — this is fatal
            raise RuntimeError(
                f"ELLM ApiKeyManager: no API key available and refresh failed "
                f"(scene_code={self._scene_code})"
            ) from e

    def refresh_key(self) -> str:
        """Fetch a new API key from the ELLM gateway.

        Returns:
            The newly obtained API key string.

        Raises:
            Exception: If the HTTP request fails or the response is invalid.
        """
        req_message = json.dumps(
            {
                "REQ_HEAD": {"TRAN_PROCESS": "", "TRAN_ID": ""},
                "REQ_BODY": {"param": {"sceneCode": self._scene_code}},
            },
            ensure_ascii=False,
        )

        logger.debug(
            "ELLM ApiKeyManager: requesting new key (scene_code=%s)",
            self._scene_code,
        )

        client = self._get_http_client()
        try:
            response = client.post(
                self._api_key_url,
                data={"REQ_MESSAGE": req_message},
                timeout=self._request_timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(
                "ELLM ApiKeyManager: HTTP request failed (scene_code=%s, url=%s, error=%s)",
                self._scene_code,
                self._api_key_url,
                e,
            )
            raise

        return self._parse_key_response(response.json())

    def _parse_key_response(self, data: dict[str, Any]) -> str:
        """Parse the ELLM API key response and update internal state.

        Expected response format::

            {
                "RSP_BODY": {
                    "result": {
                        "apiKey": "...",
                        "timeToLive": 1776070825782
                    }
                },
                "RSP_HEAD": {
                    "TRAN_SUCCESS": "1"
                }
            }

        Returns:
            The API key string.
        """
        rsp_head = data.get("RSP_HEAD", {})
        if rsp_head.get("TRAN_SUCCESS") != "1":
            raise ValueError(
                f"ELLM ApiKeyManager: key request failed (TRAN_SUCCESS != 1, "
                f"scene_code={self._scene_code}, response={data})"
            )

        rsp_body = data.get("RSP_BODY", {})
        result = rsp_body.get("result", {})
        api_key = result.get("apiKey")
        time_to_live = result.get("timeToLive")

        if not api_key:
            raise ValueError(
                f"ELLM ApiKeyManager: no apiKey in response "
                f"(scene_code={self._scene_code}, response={data})"
            )

        with self._lock:
            self._current_key = api_key
            self._key_obtained_at = time.time()
            self._key_ttl_ms = int(time_to_live) if time_to_live else 0

        logger.info(
            "ELLM ApiKeyManager: key refreshed successfully (scene_code=%s, "
            "ttl_ms=%s, api_key=%s)",
            self._scene_code,
            self._key_ttl_ms,
            api_key,
        )
        return api_key

    def _is_near_expiry(self) -> bool:
        """Check if the current key is near its expiry time.

        Must be called while holding self._lock.

        Returns:
            True if the key is within ``refresh_ahead`` seconds of expiry.
        """
        if not self._current_key:
            return True

        if self._key_ttl_ms <= 0:
            # No TTL info — assume key is valid for refresh_interval seconds
            effective_ttl_s = self._refresh_interval
        else:
            effective_ttl_s = self._key_ttl_ms / 1000.0

        elapsed = time.time() - self._key_obtained_at
        remaining = effective_ttl_s - elapsed
        return remaining < self._refresh_ahead

    def _get_http_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.Client()
        return self._http_client

    def _refresh_loop(self) -> None:
        """Background refresh loop that periodically fetches a new key."""
        while not self._stop_event.is_set():
            # Sleep for (refresh_interval - refresh_ahead) seconds
            sleep_duration = max(self._refresh_interval - self._refresh_ahead, 60)
            if self._stop_event.wait(timeout=sleep_duration):
                break

            try:
                self.refresh_key()
            except Exception as e:
                logger.error(
                    "ELLM ApiKeyManager: background refresh failed "
                    "(scene_code=%s, error=%s), will retry in next cycle",
                    self._scene_code,
                    e,
                )

    # --- Testing helpers ---

    @classmethod
    def _reset_instances(cls) -> None:
        """Clear all singleton instances. For testing only."""
        with cls._class_lock:
            for instance in cls._instances.values():
                try:
                    instance.stop()
                except Exception:
                    pass
            cls._instances.clear()

    def _set_key_for_testing(self, key: str, ttl_ms: int = 0) -> None:
        """Set the API key directly for testing purposes."""
        with self._lock:
            self._current_key = key
            self._key_obtained_at = time.time()
            self._key_ttl_ms = ttl_ms
