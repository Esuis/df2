"""Personal Search 配置模块。

本模块提供 personal_search 工具专用的配置解析逻辑。
与 vector_search 使用不同的 API 格式（multipart/form-data + REQ_MESSAGE），
因此独立于 VectorSearchConfig。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from deerflow.config import get_app_config

DEFAULT_PERSONAL_SEARCH_REPOSITORY = "personal-search"
DEFAULT_PERSONAL_SEARCH_SOURCE_TYPE = "WDZS"
DEFAULT_PERSONAL_SEARCH_SEARCH_TYPE = "0"
DEFAULT_PERSONAL_SEARCH_TIMEOUT = 30

DEFAULT_PERSONAL_SEARCH_HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "User-Agent": "DeerFlow-PersonalSearch/2.0",
    "jumpCloud-Env": "BASE",
}


@dataclass(frozen=True)
class PersonalSearchConfig:
    """Resolved personal search configuration."""

    api_url: str
    timeout: int
    source_type: str
    repository: str
    search_type: str
    headers: dict[str, str]


def _get_personal_search_section() -> dict[str, Any]:
    """从顶层配置中获取 personal_search 配置段。"""
    app_config = get_app_config()
    if app_config.model_extra is None:
        return {}

    for key in ("personal_search", "PERSONAL_SEARCH"):
        section = app_config.model_extra.get(key)
        if isinstance(section, dict):
            return section
    return {}


def _get_tool_extra(tool_name: str = "personal_search") -> dict[str, Any]:
    """从工具配置中获取 extra 字段。"""
    tool_config = get_app_config().get_tool_config(tool_name)
    if tool_config is None or tool_config.model_extra is None:
        return {}
    return dict(tool_config.model_extra)


def _merge_dict_values(*values: Any) -> dict[str, str]:
    """合并多个字典，后面的值覆盖前面的。"""
    merged: dict[str, str] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, item in value.items():
            if item is None:
                continue
            merged[str(key)] = str(item)
    return merged


def get_personal_search_config(tool_name: str = "personal_search") -> PersonalSearchConfig:
    """解析 personal_search 配置，优先级：tool config > 顶层 config > 环境变量 > 默认值。"""

    section = _get_personal_search_section()
    tool_extra = _get_tool_extra(tool_name)

    api_url = str(
        tool_extra.get("api_url")
        or section.get("api_url")
        or os.getenv("PERSONAL_SEARCH_API_URL", "")
    )

    source_type = str(
        tool_extra.get("source_type")
        or section.get("source_type")
        or DEFAULT_PERSONAL_SEARCH_SOURCE_TYPE
    )

    repository = str(
        tool_extra.get("repository")
        or section.get("repository")
        or DEFAULT_PERSONAL_SEARCH_REPOSITORY
    )

    search_type = str(
        tool_extra.get("search_type")
        or section.get("search_type")
        or DEFAULT_PERSONAL_SEARCH_SEARCH_TYPE
    )

    timeout_value = tool_extra.get("timeout", section.get("timeout", DEFAULT_PERSONAL_SEARCH_TIMEOUT))

    # 构建 headers：默认 + 配置合并
    merged_headers = _merge_dict_values(
        DEFAULT_PERSONAL_SEARCH_HEADERS,
        section.get("headers"),
        tool_extra.get("headers"),
    )

    return PersonalSearchConfig(
        api_url=api_url,
        timeout=int(timeout_value),
        source_type=source_type,
        repository=repository,
        search_type=search_type,
        headers=merged_headers,
    )


__all__ = ["PersonalSearchConfig", "get_personal_search_config"]
