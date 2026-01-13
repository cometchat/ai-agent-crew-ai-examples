from __future__ import annotations

from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class MessagePayload(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[MessagePayload]] = None


class AgentStreamRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    thread_id: str = Field(default_factory=lambda: f"thread_{uuid4().hex[:8]}", alias="threadId")
    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex[:8]}", alias="runId")
    messages: List[MessagePayload]
