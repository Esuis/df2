import logging
import time
from typing import override
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class LLMLoggerCallback(BaseCallbackHandler):
    """LLM 调用日志回调，记录完整的输入输出内容。
    
    流式输出的内容会在 on_llm_end() 中一次性打印，而不是流式逐段打印。
    
    使用 run_id（每次 LLM 调用唯一）作为内部追踪 key，
    确保并发调用时 start_time 和累积输出不会互相覆盖。
    无 run_id 时回退到 model_name 作 key（可通过 response.llm_output 回溯）。
    """
    
    def __init__(self):
        # key 优先为 run_id (UUID)，回退为 model_name (str)
        self._start_times: dict[UUID | str, float] = {}
        self._accumulated_outputs: dict[UUID | str, list[str]] = {}
        self._model_names: dict[UUID | str, str] = {}

    @staticmethod
    def _get_thread_id() -> str | None:
        """从 LangGraph 运行时 config 中获取当前 thread_id
        
        基于 contextvars 实现，并发安全：每个协程/线程看到自己的 config。
        """
        try:
            from langgraph.config import get_config
            config = get_config()
            return config.get("configurable", {}).get("thread_id")
        except Exception:
            return None

    @override
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        """LLM 调用开始时记录输入"""
        run_id = kwargs.get("run_id")
        model_name = serialized.get("name", "unknown")

        if run_id is not None:
            # 正常路径：run_id 唯一标识此次调用
            key = run_id
        else:
            # 回退：无 run_id 时用 model_name 作 key（可通过 response.llm_output 回溯）
            key = model_name

        logger.info(f"LLM 调用开始 run_id：{key}")

        self._start_times[key] = time.time()
        self._accumulated_outputs[key] = []
        self._model_names[key] = model_name

        
        # 记录输入内容
        input_content = "\n".join(prompts) if prompts else ""
        thread_id = self._get_thread_id()
        
        logger.info("llm.invoke.start", extra={
            "thread_id": thread_id or "unknown",
            "model_name": model_name,
            "prompts_count": len(prompts),
            "input_content": input_content,
        })

    @override
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """流式输出时累积 token，但不打印日志"""
        run_id = kwargs.get("run_id")
        if run_id is not None and run_id in self._accumulated_outputs:
            # 正常路径：精确匹配 run_id
            self._accumulated_outputs[run_id].append(token)
        else:
            # 回退：遍历所有缓冲区（兼容无 run_id 的场景）
            for key in self._accumulated_outputs:
                self._accumulated_outputs[key].append(token)

    @override
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 调用结束时记录完整输出"""
        run_id = kwargs.get("run_id")
        
        # 1. 优先用 run_id 精确匹配
        start_time = self._start_times.pop(run_id, None) if run_id is not None else None
        accumulated = self._accumulated_outputs.pop(run_id, []) if run_id is not None else []
        model_name = self._model_names.pop(run_id, None) if run_id is not None else None

        # 2. 回退：用 response.llm_output 中的 model_name 匹配
        if start_time is None:
            fallback_name = None
            if response.llm_output and "model_name" in response.llm_output:
                fallback_name = response.llm_output.get("model_name")
            
            if fallback_name and fallback_name in self._start_times:
                start_time = self._start_times.pop(fallback_name)
                accumulated = self._accumulated_outputs.pop(fallback_name, [])
                model_name = self._model_names.pop(fallback_name, None)
        
        # 3. 最后回退：使用任意一个剩余条目
        if start_time is None and self._start_times:
            key = next(iter(self._start_times))
            start_time = self._start_times.pop(key)
            accumulated = self._accumulated_outputs.pop(key, [])
            model_name = self._model_names.pop(key, None)
        
        duration_ms = (time.time() - start_time) * 1000 if start_time else 0
        token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        
        # 获取完整的输出内容
        if accumulated:
            output_content = "".join(accumulated)
        else:
            # 非流式响应，从 generations 中提取
            output_content = ""
            if response.generations:
                for generation_list in response.generations:
                    for generation in generation_list:
                        if hasattr(generation, 'text'):
                            output_content += generation.text
                        elif hasattr(generation, 'message') and generation.message:
                            output_content += str(generation.message.content)
        
        thread_id = self._get_thread_id()
        
        logger.info("llm.invoke.end", extra={
            "thread_id": thread_id or "unknown",
            "model_name": model_name or "unknown",
            "duration_ms": round(duration_ms, 2),
            "prompt_tokens": token_usage.get("prompt_tokens"),
            "completion_tokens": token_usage.get("completion_tokens"),
            "total_tokens": token_usage.get("total_tokens"),
            "output_content": output_content,
        })

    @override
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """LLM 调用出错时记录错误"""
        run_id = kwargs.get("run_id")
        
        # 清理该调用对应的缓冲区
        if run_id is not None:
            self._start_times.pop(run_id, None)
            self._accumulated_outputs.pop(run_id, None)
            self._model_names.pop(run_id, None)
        else:
            # 回退：清理所有缓冲区（无法确定是哪次调用出错）
            self._start_times.clear()
            self._accumulated_outputs.clear()
            self._model_names.clear()
        
        thread_id = self._get_thread_id()
        
        logger.error("llm.invoke.error", extra={
            "thread_id": thread_id or "unknown",
            "error": str(error),
        })
