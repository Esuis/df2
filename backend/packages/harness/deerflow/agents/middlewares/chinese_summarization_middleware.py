"""中文版 SummarizationMiddleware，覆盖摘要前缀为中文，并增加详细的执行日志。"""

import logging
import math
from functools import partial
from typing import Any

from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class ChineseSummarizationMiddleware(SummarizationMiddleware):
    """与 LangChain SummarizationMiddleware 完全一致，仅将摘要前缀改为中文，并增加执行日志。

    Log event 前缀约定：
      summarization.before_model.{enter|exit}
      summarization.should_summarize.{check_condition|result}
      summarization.determine_cutoff_index
      summarization.partition
      summarization.trim_for_summary
      summarization.create_summary
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # 中文场景为主，将 chars_per_token 从默认 4.0 调整为 2.0
        self.token_counter = partial(count_tokens_approximately, chars_per_token=1.7)
        original = self.token_counter
        self.token_counter = self._make_logging_counter(original)

    # ── before_model / abefore_model — 入口和出口 ──────────────────────────

    @staticmethod
    def _get_last_usage(messages: list[AnyMessage]) -> dict[str, Any]:
        """从最新一条 AIMessage 中提取 usage_metadata。"""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.usage_metadata:
                return {
                    "input_tokens": msg.usage_metadata.get("input_tokens"),
                    "output_tokens": msg.usage_metadata.get("output_tokens"),
                    "total_tokens": msg.usage_metadata.get("total_tokens"),
                }
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        total_tokens = self.token_counter(messages)
        usage = self._get_last_usage(messages)
        logger.debug(
            "event=summarization.before_model.enter "
            "total_tokens=%d message_count=%d ",
            total_tokens,
            len(messages),
        )
        result = super().before_model(state, runtime)
        if result is None:
            logger.debug(
                "event=summarization.before_model.exit "
                "action=skipped total_tokens=%d",
                total_tokens,
            )
        else:
            new_msgs = result.get("messages", [])
            logger.debug(
                "event=summarization.before_model.exit "
                "action=summarized new_message_count=%d",
                len(new_msgs),
            )
        return result

    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        total_tokens = self.token_counter(messages)
        usage = self._get_last_usage(messages)
        logger.info(
            "event=summarization.abefore_model.enter "
            "total_tokens=%d message_count=%d ",
            total_tokens,
            len(messages),
        )
        result = await super().abefore_model(state, runtime)
        if result is None:
            logger.debug(
                "event=summarization.abefore_model.exit "
                "action=skipped total_tokens=%d",
                total_tokens,
            )
        else:
            new_msgs = result.get("messages", [])
            logger.debug(
                "event=summarization.abefore_model.exit "
                "action=summarized new_message_count=%d",
                len(new_msgs),
            )
        return result

    # ── _should_summarize — trigger 判断详情 ──────────────────────────────

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        if not self._trigger_conditions:
            logger.debug(
                "event=summarization.should_summarize "
                "result=skipped reason=no_trigger_conditions "
                "total_tokens=%d message_count=%d",
                total_tokens,
                len(messages),
            )
            return False

        for kind, value in self._trigger_conditions:
            if kind == "messages":
                logger.debug(
                    "event=summarization.should_summarize.check_condition "
                    'condition=("messages", %d) message_count=%d threshold_met=%s',
                    value,
                    len(messages),
                    len(messages) >= value,
                )
            elif kind == "tokens":
                logger.debug(
                    "event=summarization.should_summarize.check_condition "
                    'condition=("tokens", %d) total_tokens=%d threshold_met=%s',
                    value,
                    total_tokens,
                    total_tokens >= value,
                )
            elif kind == "fraction":
                max_input_tokens = self._get_profile_limits()
                threshold = int(max_input_tokens * value) if max_input_tokens else None
                logger.debug(
                    "event=summarization.should_summarize.check_condition "
                    'condition=("fraction", %s) max_input_tokens=%s threshold=%s total_tokens=%d threshold_met=%s',
                    value,
                    max_input_tokens,
                    threshold,
                    total_tokens,
                    total_tokens >= threshold if threshold else "unknown",
                )

        result = super()._should_summarize(messages, total_tokens)
        matched_condition = None
        if result and self._trigger_conditions:
            for kind, value in self._trigger_conditions:
                if kind == "messages" and len(messages) >= value:
                    matched_condition = (kind, value)
                    break
                if kind == "tokens" and total_tokens >= value:
                    matched_condition = (kind, value)
                    break
                if kind == "fraction":
                    max_input_tokens = self._get_profile_limits()
                    if max_input_tokens and total_tokens >= int(max_input_tokens * value):
                        matched_condition = (kind, value)
                        break

        logger.debug(
            "event=summarization.should_summarize.result "
            "result=%s total_tokens=%d message_count=%d matched_condition=%s",
            result,
            total_tokens,
            len(messages),
            matched_condition,
        )
        return result

    # ── _determine_cutoff_index — 截断位置计算 ────────────────────────────

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        result = super()._determine_cutoff_index(messages)
        kind, value = self.keep
        logger.debug(
            "event=summarization.determine_cutoff_index "
            'keep=("%s", %s) cutoff_index=%d total_messages=%d',
            kind,
            value,
            result,
            len(messages),
        )
        return result

    # ── _partition_messages — 分区统计 ────────────────────────────────────

    def _partition_messages(
        self,
        conversation_messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        to_summarize, preserved = super()._partition_messages(conversation_messages, cutoff_index)
        total_tokens = self.token_counter(preserved)
        logger.debug(
            "event=summarization.partition "
            "cutoff_index=%d to_summarize=%d preserved=%d total=%d preserved_token_estimate=%d",
            cutoff_index,
            len(to_summarize),
            len(preserved),
            len(conversation_messages),
            total_tokens,
        )
        return to_summarize, preserved

    # ── _trim_messages_for_summary — trim 过程 ────────────────────────────

    def _trim_messages_for_summary(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        result = super()._trim_messages_for_summary(messages)
        logger.debug(
            "event=summarization.trim_for_summary "
            "before=%d after=%d trim_token_limit=%s",
            len(messages),
            len(result) if result else 0,
            self.trim_tokens_to_summarize,
        )
        return result

    # ── _create_summary / _acreate_summary — 摘要生成 ──────────────────────

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        result = super()._create_summary(messages_to_summarize)
        logger.debug(
            "event=summarization.create_summary "
            "messages=%d summary_length=%d summary_preview=\"%s\"",
            len(messages_to_summarize),
            len(result),
            result[:80],
        )
        return result

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        result = await super()._acreate_summary(messages_to_summarize)
        logger.debug(
            "event=summarization.acreate_summary "
            "messages=%d summary_length=%d summary_preview=\"%s\"",
            len(messages_to_summarize),
            len(result),
            result[:80],
        )
        return result

    # ── _make_logging_counter — token 计算逐条日志包装 ─────────────────────

    def _make_logging_counter(
        self, original: Any
    ) -> Any:
        """包装原始的 token_counter，逐条打印每条消息的 token 估算明细。"""
        # 从 partial 中取出 chars_per_token（Claude 模型的参数）
        if isinstance(original, partial) and "chars_per_token" in original.keywords:
            chars_per_token = original.keywords["chars_per_token"]
        else:
            chars_per_token = 4.0

        def wrapped(messages: Any) -> int:
            total = 0.0
            for i, msg in enumerate(messages):
                # 1. content 字符数
                if isinstance(msg.content, str):
                    content_len = len(msg.content)
                elif isinstance(msg.content, list):
                    # 多模态：只计文本块，图片块忽略（图片固定 85 tokens 但由 original 处理）
                    content_len = 0
                    for block in msg.content:
                        if isinstance(block, str):
                            content_len += len(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content_len += len(block.get("text", ""))
                        else:
                            content_len += len(repr(block))
                else:
                    content_len = len(repr(msg.content))

                # 2. role 字符数
                role = msg.type  # "human", "ai", "tool", "system"
                role_len = len(role)

                # 3. tool_call_id（仅 ToolMessage）
                tool_call_id_len = 0
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    tool_call_id_len = len(msg.tool_call_id)

                # 4. AIMessage 的 tool_calls（非 Anthropic 格式时）
                tool_calls_len = 0
                if (
                    hasattr(msg, "tool_calls")
                    and msg.tool_calls
                    and not isinstance(msg.content, list)
                ):
                    tool_calls_len = len(repr(msg.tool_calls))

                # 5. name
                name_len = 0
                if msg.name:
                    name_len = len(msg.name)

                msg_chars = content_len + role_len + tool_call_id_len + tool_calls_len + name_len
                msg_tokens = math.ceil(msg_chars / chars_per_token) + 3  # extra_tokens_per_message
                total += msg_tokens

                logger.debug(
                    "token_counter msg[%d] type=%-6s "
                    "content_len=%d role_len=%d tool_call_id_len=%d "
                    "tool_calls_len=%d name_len=%d "
                    "msg_chars=%d msg_tokens=%d cumulative=%.1f",
                    i,
                    msg.type,
                    content_len,
                    role_len,
                    tool_call_id_len,
                    tool_calls_len,
                    name_len,
                    msg_chars,
                    msg_tokens,
                    total,
                )

            final = math.ceil(total)
            logger.debug(
                "token_counter result total=%d messages=%d chars_per_token=%s original_result=%d",
                final,
                len(messages),
                chars_per_token,
                original(messages),
            )
            return final

        return wrapped

    # ── _build_new_messages — 中文摘要前缀 ─────────────────────────────────

    def _build_new_messages(self, summary: str) -> list[HumanMessage]:
        return [
            HumanMessage(content=f"以下是对话历史摘要：\n\n{summary}")
        ]
