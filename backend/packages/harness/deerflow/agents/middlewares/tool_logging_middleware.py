"""ToolLoggingMiddleware - 统一拦截所有工具调用并添加日志的中间件。"""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)


class ToolLoggingMiddleware(AgentMiddleware[ThreadState]):
    """统一拦截所有工具调用并记录完整输入输出的中间件。
    
    特性：
    - 零侵入：无需修改任何现有 tool 代码
    - 全覆盖：自动覆盖所有工具（bash、read_file、task、MCP 工具等）
    - 统一格式：所有工具日志格式一致，便于分析
    - 性能统计：自动记录调用耗时
    """

    state_schema = ThreadState

    def _log_tool_call(self, request: ToolCallRequest, phase: str, **extra):
        """记录工具调用日志"""
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})
        
        logger.info(f"tool.{phase}", extra={
            "tool_name": tool_name,
            "tool_args": tool_args,
            **extra
        })

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """同步工具调用包装器"""
        start_time = time.time()
        
        # 记录调用开始
        self._log_tool_call(request, "invoke.start")
        
        try:
            result = handler(request)
            duration_ms = (time.time() - start_time) * 1000
            
            # 提取结果内容
            result_content = ""
            if isinstance(result, ToolMessage):
                result_content = str(result.content) if result.content else ""
            
            # 记录调用成功结束
            self._log_tool_call(request, "invoke.end",
                duration_ms=round(duration_ms, 2),
                result_content=result_content
            )
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # 记录调用失败
            self._log_tool_call(request, "invoke.error",
                duration_ms=round(duration_ms, 2),
                error=str(e)
            )
            raise

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """异步工具调用包装器"""
        start_time = time.time()
        
        # 记录调用开始
        self._log_tool_call(request, "invoke.start")
        
        try:
            result = await handler(request)
            duration_ms = (time.time() - start_time) * 1000
            
            # 提取结果内容
            result_content = ""
            if isinstance(result, ToolMessage):
                result_content = str(result.content) if result.content else ""
            
            # 记录调用成功结束
            self._log_tool_call(request, "invoke.end",
                duration_ms=round(duration_ms, 2),
                result_content=result_content
            )
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # 记录调用失败
            self._log_tool_call(request, "invoke.error",
                duration_ms=round(duration_ms, 2),
                error=str(e)
            )
            raise
