import logging
import time
from typing import override
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class LLMLoggerCallback(BaseCallbackHandler):
    """LLM 调用日志回调，记录完整的输入输出内容。
    
    流式输出的内容会在 on_llm_end() 中一次性打印，而不是流式逐段打印。
    """
    
    def __init__(self):
        self._start_times = {}
        self._accumulated_outputs = {}  # 用于累积流式输出内容



    @override
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        """LLM 调用开始时记录输入"""
        # 使用模型名称作为 key 来跟踪多个并行的 LLM 调用
        model_name = serialized.get("name", "unknown")
        self._start_times[model_name] = time.time()
        self._accumulated_outputs[model_name] = []  # 初始化累积缓冲区
        
        # 记录输入内容
        input_content = "\n".join(prompts) if prompts else ""
        
        logger.info("llm.invoke.start", extra={
            "model_name": model_name,
            "prompts_count": len(prompts),
            "input_content": input_content,
        })

    @override
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """流式输出时累积 token，但不打印日志"""
        # 由于无法确定是哪个模型产生的 token，需要遍历所有累积缓冲区
        # 在实际场景中，通常只有一个活跃的 LLM 调用
        for key in self._accumulated_outputs:
            self._accumulated_outputs[key].append(token)

    @override
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 调用结束时记录完整输出"""
        # 获取模型名称
        model_name = None
        if response.llm_output and "model_name" in response.llm_output:
            model_name = response.llm_output.get("model_name")
        
        # 尝试找到对应的 start_time 和 accumulated output
        start_time = None
        accumulated = []
        if model_name and model_name in self._start_times:
            start_time = self._start_times.pop(model_name)
            accumulated = self._accumulated_outputs.pop(model_name, [])
        else:
            # 如果没有匹配到，使用任意一个（通常只有一个）
            if self._start_times:
                key = next(iter(self._start_times))
                start_time = self._start_times.pop(key)
                accumulated = self._accumulated_outputs.pop(key, [])
        
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
        
        logger.info("llm.invoke.end", extra={
            "duration_ms": round(duration_ms, 2),
            "prompt_tokens": token_usage.get("prompt_tokens"),
            "completion_tokens": token_usage.get("completion_tokens"),
            "total_tokens": token_usage.get("total_tokens"),
            "output_content": output_content,
        })

    @override
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """LLM 调用出错时记录错误"""
        # 清理所有累积缓冲区（通常只有一个活跃的调用）
        self._start_times.clear()
        self._accumulated_outputs.clear()
        
        logger.error("llm.invoke.error", extra={
            "error": str(error),
        })
