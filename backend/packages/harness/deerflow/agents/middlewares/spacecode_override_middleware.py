"""SpacecodeOverrideMiddleware - 拦截 personal_search 工具调用，用文件中的 space_code 强制覆盖 space_code 参数。

当模型调用 personal_search 时，有时会填充错误的 space_code。
本中间件在工具执行前从线程目录下的 custom_params.json 文件中读取 space_code，
覆盖模型生成的 space_code 参数，确保搜索请求使用正确的知识空间代码。

文件路径: {base_dir}/threads/{thread_id}/custom_params.json
文件格式: {"vector_search_switch": true, "online_search_switch": false, "space_code": ["SP0999999"]}
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.thread_state import ThreadState
from deerflow.config.paths import get_paths

logger = logging.getLogger(__name__)

_TARGET_TOOL_NAME = "personal_search"
_CUSTOM_PARAMS_FILENAME = "custom_params.json"


class SpacecodeOverrideMiddleware(AgentMiddleware[ThreadState]):
    """拦截 personal_search 工具调用，用文件中的 space_code 覆盖 space_code。

    数据流:
        线程目录 custom_params.json 文件
          → get_paths().thread_dir(thread_id) / "custom_params.json"
            → 本中间件读取 space_code 并覆盖 tool_call.args["space_code"]
    """

    state_schema = ThreadState

    @staticmethod
    def _get_thread_id(request: ToolCallRequest) -> str | None:
        """从运行时上下文中获取 thread_id。

        优先从 runtime.context 获取，回退到 runtime.config.configurable。
        """
        runtime = request.runtime
        if runtime is None:
            return None
        ctx = getattr(runtime, "context", None) or {}
        thread_id = ctx.get("thread_id") if isinstance(ctx, dict) else None
        if thread_id is None:
            cfg = getattr(runtime, "config", None) or {}
            thread_id = cfg.get("configurable", {}).get("thread_id")
        return thread_id

    @staticmethod
    def _read_space_code_from_file(thread_id: str) -> list[str] | None:
        """从线程目录的 custom_params.json 文件中读取 space_code。

        Args:
            thread_id: 线程 ID，用于定位线程目录。

        Returns:
            space_code 列表（统一为 list[str]），文件不存在或无 space_code 时返回 None。
        """
        try:
            paths = get_paths()
            params_file = paths.thread_dir(thread_id) / _CUSTOM_PARAMS_FILENAME
        except (ValueError, OSError) as exc:
            logger.warning("SpacecodeOverride: 无法构建线程目录路径 thread_id=%s: %s", thread_id, exc)
            return None

        if not params_file.is_file():
            logger.debug("SpacecodeOverride: 文件不存在 %s，跳过覆盖", params_file)
            return None

        try:
            raw = params_file.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("SpacecodeOverride: 读取或解析文件失败 %s: %s", params_file, exc)
            return None

        space_code = data.get("space_code")
        if space_code is None:
            logger.debug("SpacecodeOverride: 文件 %s 中无 space_code 字段，跳过覆盖", params_file)
            return None

        # space_code 可以是 str 或 list[str]，统一转为 list[str]
        if isinstance(space_code, str):
            return [space_code]
        if isinstance(space_code, list):
            return space_code

        logger.warning(
            "SpacecodeOverride: space_code 类型不支持: %s，期望 str 或 list[str]，跳过覆盖",
            type(space_code).__name__,
        )
        return None

    def _maybe_override(self, request: ToolCallRequest) -> None:
        """若工具为 personal_search 且文件中存在有效 space_code，则覆盖参数。"""
        tool_name = request.tool_call.get("name")
        if tool_name != _TARGET_TOOL_NAME:
            return

        thread_id = self._get_thread_id(request)
        if not thread_id:
            logger.debug("SpacecodeOverride: 无法获取 thread_id，跳过覆盖")
            return

        normalised = self._read_space_code_from_file(thread_id)
        if normalised is None:
            return

        args = request.tool_call.setdefault("args", {})
        original = args.get("space_code")
        args["space_code"] = normalised
        logger.info(
            "SpacecodeOverride: thread_id=%s tool_name=%s 覆盖 space_code: %s → %s",
            thread_id,
            tool_name,
            original,
            normalised,
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        self._maybe_override(request)
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        self._maybe_override(request)
        return await handler(request)
