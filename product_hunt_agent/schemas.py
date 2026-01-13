from __future__ import annotations

from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MessagePayload(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[MessagePayload]] = None


class AgentStreamRequest(BaseModel):
    thread_id: str = Field(default_factory=lambda: f"thread_{uuid4().hex[:8]}")
    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex[:8]}")
    messages: List[MessagePayload]
