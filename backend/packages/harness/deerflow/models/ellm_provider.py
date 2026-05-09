"""ELLM ChatOpenAI Provider with dynamic API key refresh.

This module provides ``EllmChatModel``, a custom ``ChatOpenAI`` subclass
designed for the BOCOM ELLM (Enterprise Large Language Model) gateway.

The ELLM gateway requires an ``api-key`` header (not the standard
``Authorization: Bearer`` header) and the key expires approximately every
25 minutes.  This provider:

1.  On initialisation, obtains an API key via the ELLM key-service endpoint
    and starts a background refresh thread.
2.  Before every LLM call, injects the latest valid key into
    ``default_headers["api-key"]`` so the request always carries a fresh key.

Configuration example (``config.yaml``)::

    models:
      - name: qwen3-235b-ellm
        display_name: Qwen3-235B-A22B (交银ELLM)
        use: deerflow.models.ellm_provider:EllmChatModel
        model: Qwen3-235B-A22B
        base_url: http://eaip-chn-slb-7006.bocomm.com/ELLM.ELLM-ADAPTER.V-1.0/v1
        api_key_url: http://eaip-ellm-1.bocomm.com/ELLM.ELLM-OMSERVICE.V-1.0/createSceneApiKey.do
        scene_code: P2024146
        api_key_refresh_interval: 1800
        api_key_refresh_ahead: 300
        max_tokens: 4096
        temperature: 0.7
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field, SecretStr
from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI

from deerflow.models.ellm_apikey_manager import EllmApiKeyManager

logger = logging.getLogger(__name__)

# Placeholder API key value to satisfy ChatOpenAI's api_key validation.
# The actual authentication is done via the "api-key" custom header.
_ELLM_PLACEHOLDER_API_KEY = "ellm-dynamic-key"


class EllmChatModel(ChatOpenAI):
    """ChatOpenAI with dynamic API key refresh for the BOCOM ELLM gateway.

    The ELLM gateway authenticates via an ``api-key`` header whose value
    expires approximately every 25 minutes.  This provider transparently
    refreshes the key in the background so that every LLM request carries
    a valid key without requiring a process restart.

    Custom configuration fields:

    - ``api_key_url``: URL of the ELLM key-service endpoint
      (e.g. ``http://eaip-ellm-1.bocomm.com/ELLM.ELLM-OMSERVICE.V-1.0/createSceneApiKey.do``)
    - ``scene_code``: Scene code for the ELLM key-service
      (e.g. ``P2024146``)
    - ``api_key_refresh_interval``: How often (in seconds) to refresh the key.
      Defaults to 1800 (30 minutes).
    - ``api_key_refresh_ahead``: How many seconds before expiry to trigger a
      refresh. Defaults to 300 (5 minutes).
    """

    # Custom configuration fields
    api_key_url: str = ""
    scene_code: str = ""
    api_key_refresh_interval: int = 1800
    api_key_refresh_ahead: int = 300

    # 必须在父类初始化校验前就存在一个占位 api_key
    openai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(_ELLM_PLACEHOLDER_API_KEY),
        alias='api_key',
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialise the API key manager and set up default headers."""
        if not self.api_key_url:
            raise ValueError(
                "EllmChatModel requires 'api_key_url' to be configured in config.yaml"
            )
        if not self.scene_code:
            raise ValueError(
                "EllmChatModel requires 'scene_code' to be configured in config.yaml"
            )

        # Get or create the singleton key manager for this scene_code
        self._key_manager = EllmApiKeyManager.get_instance(
            api_key_url=self.api_key_url,
            scene_code=self.scene_code,
            refresh_interval=self.api_key_refresh_interval,
            refresh_ahead=self.api_key_refresh_ahead,
        )

        # Start the manager (initial key fetch + background refresh thread)
        self._key_manager.start()

        # Obtain initial key
        current_key = self._key_manager.get_api_key()

        # Inject the real key as a custom header.
        # The ELLM gateway expects "api-key" header, not "Authorization: Bearer".
        self.default_headers = {
            **(self.default_headers or {}),
            "api-key": current_key,
        }

        logger.info(
            "EllmChatModel initialised (scene_code=%s, model=%s)",
            self.scene_code,
            self.model_name,
        )

        super().model_post_init(__context)

    def _inject_latest_api_key(self) -> None:
        """Update default_headers with the latest API key from the manager."""
        try:
            current_key = self._key_manager.get_api_key()
            self.default_headers = {
                **(self.default_headers or {}),
                "api-key": current_key,
            }
        except Exception as e:
            logger.warning(
                "EllmChatModel: failed to refresh api-key header, "
                "existing header will be used (error=%s)",
                e,
            )

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Inject the latest API key into the request payload before sending."""
        self._inject_latest_api_key()
        logger.info(
            "ELLM request headers: %s",
            self.default_headers,
        )
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return payload