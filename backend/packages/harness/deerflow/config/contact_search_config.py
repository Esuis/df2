from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from deerflow.config import get_app_config


@dataclass(frozen=True)
class ContactSearchConfig:
    """Resolved contact search configuration."""

    api_url: str
    timeout: int
    headers: dict[str, str]
    cookies: dict[str, str]


def _get_contact_search_section() -> dict[str, Any]:
    app_config = get_app_config()
    if app_config.model_extra is None:
        return {}

    for key in ("contact_search", "CONTACT_SEARCH"):
        section = app_config.model_extra.get(key)
        if isinstance(section, dict):
            return section
    return {}


def _get_tool_extra(tool_name: str) -> dict[str, Any]:
    tool_config = get_app_config().get_tool_config(tool_name)
    if tool_config is None or tool_config.model_extra is None:
        return {}
    return dict(tool_config.model_extra)


def _merge_dict_values(*values: Any) -> dict[str, str]:
    merged: dict[str, str] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, item in value.items():
            if item is None:
                continue
            merged[str(key)] = str(item)
    return merged


def get_contact_search_config(tool_name: str = "通讯录查询") -> ContactSearchConfig:
    """Resolve the contact search config from tool config, top-level config, and env vars."""

    section = _get_contact_search_section()
    tool_extra = _get_tool_extra(tool_name)

    api_url = str(
        tool_extra.get("api_url")
        or section.get("api_url")
        or os.getenv("CONTACT_SEARCH_API_URL", "")
    )
    timeout_value = tool_extra.get(
        "timeout",
        section.get("timeout", os.getenv("CONTACT_SEARCH_TIMEOUT", 30)),
    )
    return ContactSearchConfig(
        api_url=api_url,
        timeout=int(timeout_value),
        headers=_merge_dict_values(
            section.get("headers"),
            tool_extra.get("headers"),
        ),
        cookies=_merge_dict_values(
            section.get("cookies"),
            tool_extra.get("cookies"),
        ),
    )


__all__ = ["ContactSearchConfig", "get_contact_search_config"]
