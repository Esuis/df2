from __future__ import annotations

import json
import logging
from typing import Optional

import requests
from langchain.tools import tool

from deerflow.config.contact_search_config import get_contact_search_config
from deerflow.community.common.auth_context import get_resolved_auth

logger = logging.getLogger(__name__)

CONTACT_SEARCH_FIELDS = [
    "userName",
    "userIds",
    "orgId",
    "orgIds",
    "orgName",
    "cellPhone",
    "telNo",
    "telNoExt",
    "shortNo",
    "email",
    "ehrPosition",
    "userPosition",
    "positionStatus",
    "customCellPhone1",
    "loginName",
]


def _build_request_body(
    userName: Optional[str] = None,
    userIds: Optional[str] = None,
    orgId: Optional[str] = None,
    orgIds: Optional[str] = None,
    orgName: Optional[str] = None,
    cellPhone: Optional[str] = None,
    telNo: Optional[str] = None,
    telNoExt: Optional[str] = None,
    shortNo: Optional[str] = None,
    email: Optional[str] = None,
    ehrPosition: Optional[str] = None,
    userPosition: Optional[str] = None,
    positionStatus: Optional[str] = None,
    customCellPhone1: Optional[str] = None,
    loginName: Optional[str] = None,
) -> dict[str, str]:
    """Build a request body containing only the non-None parameters."""
    body: dict[str, str] = {}
    local_vars = {
        "userName": userName,
        "userIds": userIds,
        "orgId": orgId,
        "orgIds": orgIds,
        "orgName": orgName,
        "cellPhone": cellPhone,
        "telNo": telNo,
        "telNoExt": telNoExt,
        "shortNo": shortNo,
        "email": email,
        "ehrPosition": ehrPosition,
        "userPosition": userPosition,
        "positionStatus": positionStatus,
        "customCellPhone1": customCellPhone1,
        "loginName": loginName,
    }
    for field in CONTACT_SEARCH_FIELDS:
        value = local_vars[field]
        if value is not None:
            body[field] = value
    return body


def search_contact_backend(
    userName: Optional[str] = None,
    userIds: Optional[str] = None,
    orgId: Optional[str] = None,
    orgIds: Optional[str] = None,
    orgName: Optional[str] = None,
    cellPhone: Optional[str] = None,
    telNo: Optional[str] = None,
    telNoExt: Optional[str] = None,
    shortNo: Optional[str] = None,
    email: Optional[str] = None,
    ehrPosition: Optional[str] = None,
    userPosition: Optional[str] = None,
    positionStatus: Optional[str] = None,
    customCellPhone1: Optional[str] = None,
    loginName: Optional[str] = None,
) -> str:
    config = get_contact_search_config()
    if not config.api_url:
        raise ValueError("通讯录查询 API URL 未配置。请在 config.yaml 或环境变量 CONTACT_SEARCH_API_URL 中设置。")

    body = _build_request_body(
        userName=userName,
        userIds=userIds,
        orgId=orgId,
        orgIds=orgIds,
        orgName=orgName,
        cellPhone=cellPhone,
        telNo=telNo,
        telNoExt=telNoExt,
        shortNo=shortNo,
        email=email,
        ehrPosition=ehrPosition,
        userPosition=userPosition,
        positionStatus=positionStatus,
        customCellPhone1=customCellPhone1,
        loginName=loginName,
    )

    if not body:
        raise ValueError("至少需要传入一个搜索参数。")

    headers = dict(config.headers)
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
        # cookies=config.cookies,
        json=body,
        timeout=config.timeout,
    )
    response.raise_for_status()
    logger.info("Contact search raw response body: %s", response.text)

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"通讯录查询返回了无效的 JSON: {exc}") from exc

    data = payload.get("data", [])
    return json.dumps(data, indent=2, ensure_ascii=False)


@tool("通讯录查询", parse_docstring=True)
def contact_search_tool(
    userName: Optional[str] = None,
    userIds: Optional[str] = None,
    orgId: Optional[str] = None,
    orgIds: Optional[str] = None,
    orgName: Optional[str] = None,
    cellPhone: Optional[str] = None,
    telNo: Optional[str] = None,
    telNoExt: Optional[str] = None,
    shortNo: Optional[str] = None,
    email: Optional[str] = None,
    ehrPosition: Optional[str] = None,
    userPosition: Optional[str] = None,
    positionStatus: Optional[str] = None,
    customCellPhone1: Optional[str] = None,
    loginName: Optional[str] = None,
) -> str:
    """根据员工姓名、手机号、邮箱等条件，在企业通讯录中查询员工信息。

    当你需要查找同事的联系方式、所属机构、职务等信息时，优先使用本工具。
    可根据实际情况传入一个或多个搜索条件，工具会自动组合条件进行查询。

    Args:
        userName: 员工姓名
        userIds: 员工ID
        orgId: 机构ID，配合 orgIds 是否级联精确匹配
        orgIds: 机构ID列表，配合 orgId 是否级联精确匹配
        orgName: 机构名称，分词匹配，不支持左右模糊
        cellPhone: 手机号，左右模糊匹配
        telNo: 座机，左右模糊匹配
        telNoExt: 座机分机，左右模糊匹配
        shortNo: 短号，左右模糊匹配
        email: 邮箱，左右模糊匹配
        ehrPosition: 职位，分词匹配，不支持左右模糊
        userPosition: 职务，精确匹配
        positionStatus: 是否兼职，精确匹配
        customCellPhone1: 用户手机号1，左右模糊匹配
        loginName: 登录名，精确匹配
    """
    try:
        return search_contact_backend(
            userName=userName,
            userIds=userIds,
            orgId=orgId,
            orgIds=orgIds,
            orgName=orgName,
            cellPhone=cellPhone,
            telNo=telNo,
            telNoExt=telNoExt,
            shortNo=shortNo,
            email=email,
            ehrPosition=ehrPosition,
            userPosition=userPosition,
            positionStatus=positionStatus,
            customCellPhone1=customCellPhone1,
            loginName=loginName,
        )
    except requests.Timeout:
        logger.error("通讯录查询请求超时。", exc_info=True)
        return json.dumps([{"error": "通讯录查询请求超时。"}], ensure_ascii=False)
    except requests.RequestException as exc:
        logger.error("通讯录查询请求失败: %s", exc, exc_info=True)
        return json.dumps([{"error": f"通讯录查询请求失败: {exc}"}], ensure_ascii=False)
    except ValueError as exc:
        return json.dumps([{"error": f"{exc}"}], ensure_ascii=False)
    except Exception as exc:
        logger.error("通讯录查询未知错误: %s", exc, exc_info=True)
        return json.dumps([{"error": f"通讯录查询失败: {exc}"}], ensure_ascii=False)


__all__ = ["contact_search_tool", "search_contact_backend"]
