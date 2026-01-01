from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel


class MessagePayload(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[MessagePayload]] = None


class AgentStreamRequest(BaseModel):
    thread_id: Optional[str] = None
    run_id: Optional[str] = None
    messages: List[MessagePayload]

