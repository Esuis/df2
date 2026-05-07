"""中文版 SummarizationMiddleware，覆盖摘要前缀为中文。"""

from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages.human import HumanMessage


class ChineseSummarizationMiddleware(SummarizationMiddleware):
    """与 LangChain SummarizationMiddleware 完全一致，仅将摘要前缀改为中文。"""

    def _build_new_messages(self, summary: str) -> list[HumanMessage]:
        return [
            HumanMessage(content=f"以下是对话历史摘要：\n\n{summary}")
        ]
