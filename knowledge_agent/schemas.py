from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class MessagePayload(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class IngestRequest(BaseModel):
    namespace: Optional[str] = None
    sources: Optional[List[Union[str, Dict[str, Any]]]] = Field(default=None)


class SearchRequest(BaseModel):
    namespace: Optional[str] = None
    query: str
    max_results: Optional[int] = Field(default=6, ge=1, le=50)


class GenerateRequest(BaseModel):
    namespace: Optional[str] = None
    messages: List[MessagePayload]


class AgentStreamRequest(BaseModel):
    thread_id: Optional[str] = None
    run_id: Optional[str] = None
    namespace: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    messages: List[MessagePayload]

