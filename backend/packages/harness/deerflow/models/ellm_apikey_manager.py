"""ELLM ApiKey Manager - Dynamic API key refresh for BOCOM ELLM gateway.

This module provides a thread-safe singleton that automatically refreshes
the API key for the BOCOM ELLM (Enterprise Large Language Model) gateway.

Key features:
  - Background thread refreshes the API key every ``refresh_interval`` seconds
  - ``get_api_key()`` returns the latest valid key, auto-refreshing if near expiry
  - Per ``scene_code`` singleton instances to avoid duplicate requests
  - Graceful degradation: keeps the old key if refresh fails
  - Cross-process shared cache via ``DEER_FLOW_HOME`` directory:
    only one process makes the HTTP request; others read the cached key
  - ``timeToLive`` auto-detection: supports both Unix timestamp (ms) and
    TTL duration (ms) formats from the ELLM API response
"""

from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default refresh configuration
DEFAULT_REFRESH_INTERVAL = 1800  # 25 minutes in seconds
DEFAULT_REFRESH_AHEAD = 300  # Refresh 5 minutes before expiry
DEFAULT_REQUEST_TIMEOUT = 30  # 30 seconds timeout for key request

# Values >= this threshold are treated as Unix timestamps (ms);
# smaller values are treated as TTL durations (ms).
# Rationale: 10^12 ms ≈ Sept 2001 — any real timestamp is far above this,
# while even a 1-year TTL (≈ 3.15 × 10^10 ms) is well below.
_TIMESTAMP_THRESHOLD_MS = 1_000_000_000_000

# Beijing timezone (UTC+8)
_BJ_TZ = timezone(timedelta(hours=8))


def _fmt_bj(ts: float) -> str:
    """Format a Unix timestamp as a human-readable Beijing time string."""
    if ts <= 0:
        return "N/A"
    return datetime.fromtimestamp(ts, tz=_BJ_TZ).strftime("%Y-%m-%d %H:%M:%S")


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
        self._key_ttl_ms: int = 0  # Raw timeToLive value from the response (for logging)
        self._key_expiry_time: float = 0.0  # Absolute Unix timestamp (seconds) when key expires
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
                "ELLM ApiKeyManager started for scene_code=%s, initial key obtained (pid=%s)",
                self._scene_code,
                os.getpid(),
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
            name=f"ellm-apikey-refresh-{self._scene_code}-pid{os.getpid()}",
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
        """Fetch a new API key, using the cross-process shared cache when possible.

        Flow:
          1. Try to read a fresh key from the shared cache file (no lock).
          2. If cache is fresh → load into memory and return.
          3. If cache is stale/missing → acquire file lock, double-check cache,
             then HTTP-refresh if still stale, and write back to cache.

        Returns:
            The current valid API key string.

        Raises:
            Exception: If the HTTP request fails or the response is invalid.
        """
        # Step 1: Fast path — read cache without lock
        cached = self._load_from_cache()
        if cached:
            return cached

        # Step 2: Cache miss or stale — acquire lock and refresh
        return self._acquire_lock_and_refresh()

    def _fetch_key_from_server(self) -> str:
        """Pure HTTP fetch — call the ELLM gateway and parse the response.

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

            # Compute absolute expiry time from timeToLive.
            # The ELLM API returns a Unix timestamp in ms (e.g. 1776070825782),
            # but some API versions might return a TTL duration in ms.
            # We detect which one it is by a simple magnitude threshold.
            ttl_int = int(time_to_live) if time_to_live else 0
            if ttl_int >= _TIMESTAMP_THRESHOLD_MS:
                # timeToLive is an absolute expiry timestamp (ms)
                self._key_expiry_time = ttl_int / 1000.0
            elif ttl_int > 0:
                # timeToLive is a TTL duration (ms) — add to current time
                self._key_expiry_time = time.time() + ttl_int / 1000.0
            else:
                # No TTL info
                self._key_expiry_time = 0.0

        logger.info(
            "ELLM ApiKeyManager: key refreshed successfully (pid=%s, scene_code=%s, "
            "ttl_ms=%s, expiry=%s, api_key=%s)",
            os.getpid(),
            self._scene_code,
            self._key_ttl_ms,
            _fmt_bj(self._key_expiry_time),
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

        if self._key_expiry_time > 0:
            # We have a known expiry time — compute remaining directly
            remaining = self._key_expiry_time - time.time()
        else:
            # No expiry info from API — assume key is valid for refresh_interval
            effective_expiry = self._key_obtained_at + self._refresh_interval
            remaining = effective_expiry - time.time()

        return remaining < self._refresh_ahead

    # --- Cross-process shared cache ---

    def _get_shared_dir(self) -> Path:
        """Return the directory used for cross-process key cache files.

        Resolution order:
          1. ``DEER_FLOW_HOME`` environment variable
          2. ``.deer-flow`` directory under the backend root (cwd)
          3. System temp directory as fallback
        """
        env_dir = os.environ.get("DEER_FLOW_HOME")
        if env_dir:
            return Path(env_dir)

        cwd_deer_flow = Path.cwd() / ".deer-flow"
        if cwd_deer_flow.is_dir():
            return cwd_deer_flow

        # Fallback: create a temp-based directory
        fallback = Path(tempfile.gettempdir()) / "deer-flow" / "ellm_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _cache_path(self) -> Path:
        """Return the JSON cache file path for this scene_code."""
        return self._get_shared_dir() / f"ellm_apikey_{self._scene_code}.json"

    def _lock_path(self) -> Path:
        """Return the lock file path for this scene_code."""
        return self._get_shared_dir() / f"ellm_apikey_{self._scene_code}.lock"

    def _cache_validity_seconds(self) -> float:
        """How many seconds a cache entry is considered fresh."""
        return max(self._refresh_interval - self._refresh_ahead, 60)

    def _read_cache(self) -> dict[str, Any] | None:
        """Read the shared cache file; return None if missing or invalid."""
        try:
            path = self._cache_path()
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            obtained_at = data.get("obtained_at", 0)
            if time.time() - obtained_at < self._cache_validity_seconds():
                return data
            return None  # stale
        except Exception:
            return None

    def _write_cache(self, api_key: str, ttl_ms: int, expiry_time: float) -> None:
        """Write the current key info to the shared cache file."""
        try:
            path = self._cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "api_key": api_key,
                "obtained_at": time.time(),
                "ttl_ms": ttl_ms,
                "expiry_time": expiry_time,
                "scene_code": self._scene_code,
                "pid": os.getpid(),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(
                "ELLM ApiKeyManager: failed to write cache file (scene_code=%s, error=%s)",
                self._scene_code,
                e,
            )

    def _load_from_cache(self) -> str | None:
        """Try to load a fresh key from the shared cache into memory.

        Returns the key string if a fresh cache was loaded, None otherwise.
        """
        cached = self._read_cache()
        if cached is None:
            return None

        api_key = cached.get("api_key", "")
        if not api_key:
            return None

        obtained_at = cached.get("obtained_at", 0)
        ttl_ms = cached.get("ttl_ms", 0)
        expiry_time = cached.get("expiry_time", 0.0)

        with self._lock:
            self._current_key = api_key
            self._key_obtained_at = obtained_at
            self._key_ttl_ms = ttl_ms
            if expiry_time > 0:
                self._key_expiry_time = expiry_time
            elif ttl_ms >= _TIMESTAMP_THRESHOLD_MS:
                self._key_expiry_time = ttl_ms / 1000.0
            elif ttl_ms > 0:
                self._key_expiry_time = obtained_at + ttl_ms / 1000.0
            else:
                self._key_expiry_time = 0.0

        logger.info(
            "ELLM ApiKeyManager: loaded key from shared cache (pid=%s, scene_code=%s, "
            "obtained_at=%s, api_key=%s)",
            os.getpid(),
            self._scene_code,
            _fmt_bj(obtained_at),
            api_key,
        )
        return api_key

    def _acquire_lock_and_refresh(self) -> str:
        """Acquire the file lock, double-check cache, then HTTP refresh if needed.

        This prevents thundering herd: multiple processes discovering a stale
        cache at the same time will serialize behind the lock; only the first
        one makes the HTTP request; the rest read the fresh cache.
        """
        lock_path = self._lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        lock_fd = open(lock_path, "w")  # noqa: SIM115
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Double-check: another process may have refreshed while we waited
            cached = self._read_cache()
            if cached is not None:
                api_key = cached.get("api_key", "")
                if api_key:
                    # Load the fresh cache into memory
                    result = self._load_from_cache()
                    if result:
                        logger.info(
                            "ELLM ApiKeyManager: another process refreshed while we waited "
                            "(pid=%s, scene_code=%s)",
                            os.getpid(),
                            self._scene_code,
                        )
                        return result

            # We are the chosen process — do the HTTP refresh
            api_key = self._fetch_key_from_server()

            # Write to shared cache for other processes
            with self._lock:
                self._write_cache(
                    api_key=self._current_key,
                    ttl_ms=self._key_ttl_ms,
                    expiry_time=self._key_expiry_time,
                )
            return api_key
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    # --- HTTP ---

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
            if ttl_ms >= _TIMESTAMP_THRESHOLD_MS:
                # Value is a Unix timestamp (ms) — same logic as _parse_key_response
                self._key_expiry_time = ttl_ms / 1000.0
            elif ttl_ms > 0:
                # Value is a TTL duration (ms)
                self._key_expiry_time = time.time() + ttl_ms / 1000.0
            else:
                self._key_expiry_time = 0.0
