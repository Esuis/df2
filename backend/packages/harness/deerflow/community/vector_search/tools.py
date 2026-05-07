from __future__ import annotations

import json
import logging
from typing import Any

import requests
from langchain.tools import tool

from deerflow.config.vector_search_config import VectorSearchConfig, get_vector_search_config
from deerflow.community.common.auth_context import get_resolved_auth

logger = logging.getLogger(__name__)


def _build_request_body(keyword: str, config: VectorSearchConfig) -> dict[str, Any]:
    auth = get_resolved_auth()
    body: dict[str, Any] = {
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
                    "spaceCodes": config.space_codes,
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
        },
    }
    if auth.auth_mode == "muwp-user" and auth.muwp_user:
        body["REQ_BODY"]["muwpUser"] = auth.muwp_user
    return body


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
        logger.warning("Vector search returned unexpected result payload: %s", all_entries)
        all_entries = []

    if not all_entries:
        return [{"info": f"未找到相关内容。关键词: {keyword}"}]

    return [_extract_entry_info(entry) for entry in all_entries]


def search_vector_backend(keyword: str, tool_name: str = "vector_search") -> str:
    config = get_vector_search_config(tool_name)
    if not config.api_url:
        raise ValueError("VECTOR_SEARCH_API_URL, PRODUCT_SEARCH_API_URL, or EUVD_API_URL is required. Set it in config.yaml or the environment.")

    headers = dict(config.headers)
    # 根据 auth context 添加认证请求头
    auth = get_resolved_auth()
    if auth.auth_mode == "guwp-token":
        headers["guwp-token"] = auth.guwp_token
    elif auth.auth_mode == "jrt-auth-code":
        headers["jrt-auth-code"] = auth.jrt_auth_code
    elif auth.auth_mode == "okic-token":
        headers["okic-token"] = auth.okic_token
        headers["okic-type"] = auth.okic_type

    response = requests.post(
        config.api_url,
        headers=headers,
        cookies=config.cookies,
        json=_build_request_body(keyword, config),
        timeout=config.timeout,
    )
    response.raise_for_status()
    logger.info("Vector search raw response body: %s", response.text)

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"vector search returned invalid JSON: {exc}") from exc

    results = _extract_results(payload, keyword)
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool("vector_search", parse_docstring=True)
def vector_search_tool(keyword: str) -> str:
    """Search structured knowledge in the dedicated internal knowledge backend.

    Use this as the first search tool for nearly all fact-finding, lookup, explanation,
    policy, product, procedure, FAQ, and domain-knowledge requests. Before using
    `online_search`, try `vector_search` first, then only fall back to `online_search`
    when this tool returns no relevant content, returns obviously insufficient content,
    or cannot answer the user's question.

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
    """

    try:
        return search_vector_backend(keyword)
    except requests.Timeout:
        logger.error("Vector search request timed out.", exc_info=True)
        return json.dumps([{"error": "vector search request timed out."}], ensure_ascii=False)
    except requests.RequestException as exc:
        logger.error("Vector search request failed: %s", exc, exc_info=True)
        return json.dumps([{"error": f"vector search request failed: {exc}"}], ensure_ascii=False)
    except ValueError as exc:
        return json.dumps([{"error": f"{exc}"}], ensure_ascii=False)
    except Exception as exc:
        logger.error("Unexpected vector search error: %s", exc, exc_info=True)
        return json.dumps([{"error": f"vector search failed: {exc}"}], ensure_ascii=False)


__all__ = ["vector_search_tool", "search_vector_backend"]
