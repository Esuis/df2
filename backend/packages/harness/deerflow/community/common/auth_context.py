"""认证上下文模块。

通过 ContextVar 在 agent 入口与各个 search tool 之间传递解析后的认证信息。
避免在每个 tool call 中重复读取 custom_params.json。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ResolvedAuth:
    """解析后的认证信息。

    auth_mode 表示最终选用的认证方式，优先级：guwp-token > jrt-auth-code > okic-token > muwp-user > none。
    """

    auth_mode: Literal["guwp-token", "jrt-auth-code", "okic-token", "muwp-user", "none"]
    guwp_token: str = ""
    jrt_auth_code: str = ""
    okic_token: str = ""
    okic_type: str = ""
    muwp_user: dict[str, str] = field(default_factory=dict)


_auth_ctx: ContextVar[ResolvedAuth] = ContextVar("resolved_auth", default=ResolvedAuth(auth_mode="none"))


def set_resolved_auth(auth: ResolvedAuth) -> None:
    """设置当前上下文的认证信息（agent 入口处调用）。"""
    _auth_ctx.set(auth)


def get_resolved_auth() -> ResolvedAuth:
    """获取当前上下文的认证信息（tool backend 中调用）。"""
    return _auth_ctx.get()
