from __future__ import annotations

import json
import logging
from typing import Any

import requests
from langchain.tools import tool

from deerflow.config.personal_search_config import PersonalSearchConfig, get_personal_search_config
from deerflow.community.common.auth_context import get_resolved_auth

logger = logging.getLogger(__name__)


def _build_request_body(
    keyword: str,
    config: PersonalSearchConfig,
    *,
    space_code_id: str | None = None,
    space_code: list[str] | None = None,
) -> dict[str, Any]:
    """构建 personal_search 请求体。

    API 格式：
        multipart/form-data，字段 REQ_MESSAGE 的值为 JSON 字符串，结构为：
        {"REQ_HEAD":{...},"REQ_BODY":{"param":{"sourceType":"WDZS","summaryQuestion":"...","repository":"...","searchType":"0","param":{"psnlSpaceCodeId":"...","psnlCategoryIdList":[...]}},"muwpUser":{...}}}

    psnlSpaceCodeId 和 psnlCategoryIdList 仅从工具参数获取，不从配置文件获取。
    """
    auth = get_resolved_auth()
    inner_param: dict[str, Any] = {}
    if space_code_id:
        inner_param["psnlSpaceCodeId"] = space_code_id
    if space_code:
        inner_param["psnlCategoryIdList"] = space_code

    body: dict[str, Any] = {
        "REQ_HEAD": {
            "TRANS_PROCESS": "",
            "TRAN_ID": "",
        },
        "REQ_BODY": {
            "param": {
                "sourceType": config.source_type,
                "summaryQuestion": keyword,
                "repository": config.repository,
                "searchType": config.search_type,
            },
        },
    }

    if auth.auth_mode == "muwp-user" and auth.muwp_user:
        body["REQ_BODY"]["muwpUser"] = auth.muwp_user

    if inner_param:
        body["REQ_BODY"]["param"]["param"] = inner_param

    return body


def _extract_entry_info(entry: dict[str, Any]) -> dict[str, Any]:
    title = str(entry.get("title") or "无标题")
    content = str(entry.get("content") or entry.get("absContent") or "")
    score_raw = entry.get("score")
    repository = str(entry.get("repository") or "")
    url = str(entry.get("url") or "")
    doc_id = str(entry.get("docId") or "")
    source_type = str(entry.get("sourceType") or "")
    know_type = str(entry.get("knowType") or "")
    create_time = str(entry.get("createTime") or "")
    update_time = str(entry.get("updateTime") or "")

    try:
        score = float(score_raw) if score_raw not in (None, "") else None
    except (TypeError, ValueError):
        score = None

    result: dict[str, Any] = {
        "title": title,
        "url": url,
        "doc_id": doc_id,
        "score": score,
        "repository": repository,
        "content": content,
    }
    if source_type:
        result["source_type"] = source_type
    if know_type:
        result["know_type"] = know_type
    if create_time:
        result["create_time"] = create_time
    if update_time:
        result["update_time"] = update_time

    return result


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


def search_personal_backend(
    keyword: str,
    tool_name: str = "personal_search",
    *,
    space_code_id: str | None = None,
    space_code: list[str] | None = None,
) -> str:
    """执行 personal search 后端请求。

    使用 multipart/form-data 格式发送 REQ_MESSAGE=<json>。
    """
    config = get_personal_search_config(tool_name)
    if not config.api_url:
        raise ValueError("PERSONAL_SEARCH_API_URL is required. Set it in config.yaml or the environment.")

    body = _build_request_body(keyword, config, space_code_id=space_code_id, space_code=space_code)

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

    # 使用 multipart/form-data 格式：REQ_MESSAGE 字段值为 JSON 字符串
    # requests 的 files 参数可以发送 multipart 表单，(None, value) 表示普通表单字段
    response = requests.post(
        config.api_url,
        headers=headers,
        data={"REQ_MESSAGE": (None, json.dumps(body, ensure_ascii=False))},
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
def personal_search_tool(keyword: str, space_code_id: str | None = None, space_code: list[str] | None = None) -> str:
    """在指定的知识空间内精准检索内部知识，支持按 space_code_id 和 space_code 限定搜索范围。

    当你需要在特定部门、产品线或业务域的知识空间中查找信息时，使用本工具。
    本工具与 vector_search 的区别：vector_search 在全部知识库中泛化检索，
    而本工具通过 space_code_id 和 space_code 参数将搜索限定在指定知识空间，结果更精准、
    噪声更少。当你已知目标知识空间时，优先使用本工具而非 vector_search。

    构造搜索关键词时，应贴近用户原始措辞，提取核心主题词，
    不要自动添加"交通银行"等泛化限定词，除非用户明确提及或确实需要消歧。

    关于 space_code_id 和 space_code 参数的强制规则：
    - 如果系统提示词中通过 <spacecode_directive> 标签指定了 space_code_id 或 space_code，
      你必须原样传入该值，禁止省略、修改或替换
    - 如果系统提示词未指定 space_code_id 或 space_code，则不要传入该参数，
      此时后端将使用配置中的默认知识空间

    Args:
        keyword: 搜索关键词，应取自用户请求中的核心主题词，如"理财产品"、
            "信用卡 积分规则"、"深圳分行 特邀活动 奖励"等简洁高信号短语。
        space_code_id: 个人知识空间代码ID（psnlSpaceCodeId），用于标识知识空间，
            如 "fcd15f6e27defb265f1e7e6b4d2333ca"。传入后将覆盖配置中的默认空间ID。
            若系统提示词中指定了 space_code_id，必须原样传入；否则不要传入。
        space_code: 知识分类ID列表（psnlCategoryIdList），用于限定搜索范围，
            如 ["CATE799337027285893"] 或 ["CATE799337027285893", "CATE753651591333637"]。
            传入后将覆盖配置中的默认分类ID列表。
            若系统提示词中指定了 space_code，必须原样传入；否则不要传入。
    """

    try:
        return search_personal_backend(keyword, space_code_id=space_code_id, space_code=space_code)
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
