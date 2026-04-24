from __future__ import annotations

import json
import logging
from typing import Any

import requests
from langchain.tools import tool

from deerflow.config.vector_search_config import VectorSearchConfig, get_vector_search_config

logger = logging.getLogger(__name__)


def _build_request_body(keyword: str, config: VectorSearchConfig, space_codes: list[str] | None = None) -> dict[str, Any]:
    final_space_codes = space_codes if space_codes is not None else config.space_codes
    return {
        "REQ_HEAD": {
            "TRANS_PROCESS": config.trans_process,
            "TRAN_ID": config.tran_id,
        },
        "REQ_BODY": {
            "param": {
                "summaryQuestion": keyword,
                "pageSize": config.page_size,
                "repository": config.repository,
                "param": {
                    "searchType": config.search_type,
                    "spaceCodes": final_space_codes,
                    "rerankFlag": config.rerank_flag,
                    "channelId": config.channel_id,
                    "textTopN": config.text_top_n,
                    "vectorTopN": config.vector_top_n,
                    "qaType": config.qa_type,
                    "matchFields": config.match_fields,
                    "knowStatus": config.know_status,
                    "onlineStatus": config.online_status,
                    "kpStatus": config.kp_status,
                },
            },
            "muwpUser": config.muwp_user,
        },
    }


def _extract_entry_info(entry: dict[str, Any]) -> dict[str, Any]:
    title = str(entry.get("title") or "无标题")
    content = str(entry.get("content") or entry.get("absContent") or "")
    score_raw = entry.get("score")
    repository = str(entry.get("repository") or "")
    url = str(entry.get("url") or "")
    doc_id = str(entry.get("docId") or "")

    try:
        score = float(score_raw) if score_raw not in (None, "") else None
    except (TypeError, ValueError):
        score = None

    return {
        "title": title,
        "url": url,
        "doc_id": doc_id,
        "score": score,
        "repository": repository,
        "content": content,
    }


def _extract_results(payload: dict[str, Any], keyword: str) -> list[dict[str, Any]]:
    response_head = payload.get("RSP_HEAD", {})
    if response_head and response_head.get("TRAN_SUCCESS") != "1":
        return [{"error": f"API返回错误: {response_head.get('PROCESS_STATUS_CODE', '未知错误')}"}]

    all_entries = payload.get("RSP_BODY", {}).get("result", [])
    if not isinstance(all_entries, list):
        logger.warning("Personal search returned unexpected result payload: %s", all_entries)
        all_entries = []

    if not all_entries:
        return [{"info": f"未找到相关内容。关键词: {keyword}"}]

    return [_extract_entry_info(entry) for entry in all_entries]


def search_personal_backend(keyword: str, tool_name: str = "personal_search", spacecode: list[str] | None = None) -> str:
    config = get_vector_search_config(tool_name)
    if not config.api_url:
        raise ValueError("VECTOR_SEARCH_API_URL, PRODUCT_SEARCH_API_URL, or EUVD_API_URL is required. Set it in config.yaml or the environment.")

    response = requests.post(
        config.api_url,
        headers=config.headers,
        cookies=config.cookies,
        json=_build_request_body(keyword, config, space_codes=spacecode),
        timeout=config.timeout,
    )
    response.raise_for_status()
    logger.info("Personal search raw response body: %s", response.text)

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"personal search returned invalid JSON: {exc}") from exc

    results = _extract_results(payload, keyword)
    return json.dumps(results, indent=2, ensure_ascii=False)



@tool("personal_search", parse_docstring=True)
def personal_search_tool(keyword: str, spacecode: list[str] | None = None) -> str:
    """Search personal knowledge in the dedicated internal knowledge backend with space code filtering.

    Use this tool when you need to search within a specific space code scope.
    This tool requires a `spacecode` parameter to narrow the search to a particular
    knowledge space, such as a specific department, product line, or business domain.

    When forming the search keyword, keep it close to the user's original wording.
    Do not automatically prepend or expand the query with broad qualifiers such as
    "交通银行" unless the user explicitly mentioned them or they are truly necessary
    to disambiguate the request. For line-of-business or internal-domain searches,
    use the core topic directly instead of adding "交通银行" by default.

    Args:
        keyword: Search keyword for the internal knowledge base. Prefer concise,
            high-signal phrases taken from the user's request, such as a topic,
            product, policy, procedure, activity, region, customer segment, or a
            combined query like "理财产品", "信用卡 积分规则", or "深圳分行 特邀活动 奖励".
        spacecode: Optional list of space codes to narrow the search scope. When
            provided, overrides the default space codes from configuration, e.g.
            ["SP0999999"] or ["SP0999999", "SP0888888"]. If not provided, the default
            space codes from config will be used.
    """

    try:
        return search_personal_backend(keyword, spacecode=spacecode)
    except requests.Timeout:
        logger.error("Personal search request timed out.", exc_info=True)
        return json.dumps([{"error": "personal search request timed out."}], ensure_ascii=False)
    except requests.RequestException as exc:
        logger.error("Personal search request failed: %s", exc, exc_info=True)
        return json.dumps([{"error": f"personal search request failed: {exc}"}], ensure_ascii=False)
    except ValueError as exc:
        return json.dumps([{"error": f"{exc}"}], ensure_ascii=False)
    except Exception as exc:
        logger.error("Unexpected personal search error: %s", exc, exc_info=True)
        return json.dumps([{"error": f"personal search failed: {exc}"}], ensure_ascii=False)


__all__ = ["personal_search_tool", "search_personal_backend"]
