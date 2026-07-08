"""Memory API router for retrieving and managing global memory data."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.gateway.internal_auth import get_trusted_internal_owner_user_id
from deerflow.agents.memory.updater import (
    clear_memory_data,
    get_memory_data,
    import_memory_data,
    reload_memory_data,
)
from deerflow.config.memory_config import get_memory_config
from deerflow.config.paths import make_safe_user_id
from deerflow.runtime.user_context import get_effective_user_id

router = APIRouter(prefix="/api", tags=["memory"])


def _resolve_memory_user_id(request: Request) -> str:
    """Resolve the memory owner for this request.

    Honors the trusted internal owner header that channel workers attach when
    acting for a connection owner, so an IM ``/memory`` command reads the bound
    owner's memory instead of the synthetic internal user. The header is only
    honored after ``AuthMiddleware`` validated the internal token (see
    ``get_trusted_internal_owner_user_id``). Browser/API callers are never
    internal, so this falls back to the normal contextvar-based effective user.

    The trusted owner header carries the *raw* owner id, so sanitize it through
    ``make_safe_user_id`` (the same normalization the channel file pipeline applies
    via ``_safe_user_id_for_run``/``prepare_user_dir_for_raw_id``). This keeps the
    memory bucket aligned with the owner's file/upload bucket and avoids a 500 when
    the raw id contains characters ``_validate_user_id`` would reject.
    """
    raw_owner = get_trusted_internal_owner_user_id(request)
    if raw_owner:
        return make_safe_user_id(raw_owner)
    return get_effective_user_id()


class ContextSection(BaseModel):
    """Model for context sections."""

    summary: str = Field(default="", description="Summary content")
    updatedAt: str = Field(default="", description="Last update timestamp")


class MemorySection(BaseModel):
    """Model for single-section memory."""

    summary: str = Field(default="", description="Memory summary")
    updatedAt: str = Field(default="", description="Last update timestamp")


class MemoryResponse(BaseModel):
    """Response model for memory data."""

    version: str = Field(default="1.0", description="Memory schema version")
    lastUpdated: str = Field(default="", description="Last update timestamp")
    memory: MemorySection = Field(default_factory=MemorySection)




class MemoryConfigResponse(BaseModel):
    """Response model for memory configuration."""

    enabled: bool = Field(..., description="Whether memory is enabled")
    storage_path: str = Field(..., description="Path to memory storage file")
    debounce_seconds: int = Field(..., description="Debounce time for memory updates")
    injection_enabled: bool = Field(..., description="Whether memory injection is enabled")
    max_injection_tokens: int = Field(..., description="Maximum tokens for memory injection")
    token_counting: str = Field(..., description="Token counting strategy for memory injection ('tiktoken' or 'char')")


class MemoryStatusResponse(BaseModel):
    """Response model for memory status."""

    config: MemoryConfigResponse
    data: MemoryResponse


@router.get(
    "/memory",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="Get Memory Data",
    description="Retrieve the current global memory data including user context, history, and facts.",
)
async def get_memory(http_request: Request) -> MemoryResponse:
    """Get the current global memory data.

    Returns:
        The current memory data with user context, history, and facts.

    Example Response:
        ```json
        {
            "version": "1.0",
            "lastUpdated": "2024-01-15T10:30:00Z",
            "memory": {
                "summary": "用户是后端开发者，使用 Python 和 LangGraph",
                "updatedAt": "2024-01-15T10:30:00Z"
            }
        }
        ```
    """
    memory_data = get_memory_data(user_id=_resolve_memory_user_id(http_request))
    return MemoryResponse(**memory_data)


@router.post(
    "/memory/reload",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="Reload Memory Data",
    description="Reload memory data from the storage file, refreshing the in-memory cache.",
)
async def reload_memory(http_request: Request) -> MemoryResponse:
    """Reload memory data from file.

    This forces a reload of the memory data from the storage file,
    useful when the file has been modified externally.

    Returns:
        The reloaded memory data.
    """
    memory_data = reload_memory_data(user_id=_resolve_memory_user_id(http_request))
    return MemoryResponse(**memory_data)


@router.delete(
    "/memory",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="Clear All Memory Data",
    description="Delete all saved memory data and reset the memory structure to an empty state.",
)
async def clear_memory(http_request: Request) -> MemoryResponse:
    """Clear all persisted memory data."""
    try:
        memory_data = clear_memory_data(user_id=_resolve_memory_user_id(http_request))
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to clear memory data.") from exc

    return MemoryResponse(**memory_data)




@router.get(
    "/memory/export",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="Export Memory Data",
    description="Export the current global memory data as JSON for backup or transfer.",
)
async def export_memory(http_request: Request) -> MemoryResponse:
    """Export the current memory data."""
    memory_data = get_memory_data(user_id=_resolve_memory_user_id(http_request))
    return MemoryResponse(**memory_data)


@router.post(
    "/memory/import",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="Import Memory Data",
    description="Import and overwrite the current global memory data from a JSON payload.",
)
async def import_memory(request: MemoryResponse, http_request: Request) -> MemoryResponse:
    """Import and persist memory data."""
    try:
        memory_data = import_memory_data(request.model_dump(), user_id=_resolve_memory_user_id(http_request))
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to import memory data.") from exc

    return MemoryResponse(**memory_data)


@router.get(
    "/memory/config",
    response_model=MemoryConfigResponse,
    summary="Get Memory Configuration",
    description="Retrieve the current memory system configuration.",
)
async def get_memory_config_endpoint() -> MemoryConfigResponse:
    """Get the memory system configuration.

    Returns:
        The current memory configuration settings.

    Example Response:
        ```json
        {
            "enabled": true,
            "storage_path": ".deer-flow/memory.json",
            "debounce_seconds": 30,
            "injection_enabled": true,
            "max_injection_tokens": 2000,
            "token_counting": "tiktoken"
        }
        ```
    """
    config = get_memory_config()
    return MemoryConfigResponse(
        enabled=config.enabled,
        storage_path=config.storage_path,
        debounce_seconds=config.debounce_seconds,
        injection_enabled=config.injection_enabled,
        max_injection_tokens=config.max_injection_tokens,
        token_counting=config.token_counting,
    )


@router.get(
    "/memory/status",
    response_model=MemoryStatusResponse,
    response_model_exclude_none=True,
    summary="Get Memory Status",
    description="Retrieve both memory configuration and current data in a single request.",
)
async def get_memory_status(http_request: Request) -> MemoryStatusResponse:
    """Get the memory system status including configuration and data.

    Returns:
        Combined memory configuration and current data.
    """
    config = get_memory_config()
    memory_data = get_memory_data(user_id=_resolve_memory_user_id(http_request))

    return MemoryStatusResponse(
        config=MemoryConfigResponse(
            enabled=config.enabled,
            storage_path=config.storage_path,
            debounce_seconds=config.debounce_seconds,
            injection_enabled=config.injection_enabled,
            max_injection_tokens=config.max_injection_tokens,
            token_counting=config.token_counting,
        ),
        data=MemoryResponse(**memory_data),
    )
