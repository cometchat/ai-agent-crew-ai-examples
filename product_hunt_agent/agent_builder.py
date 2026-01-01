from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from crewai import Agent, Crew, Process, Task
from crewai_tools import tool
from langchain_openai import ChatOpenAI

from .config import ProductHuntSettings
from .services import (
    get_top_products_by_timeframe,
    get_top_products_by_votes,
    get_top_products_this_week,
    parse_timeframe,
    search_products,
)


def _clamp(number: int, minimum: int, maximum: int, fallback: int) -> int:
    try:
        value = int(number)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, value))


def build_system_prompt(settings: ProductHuntSettings) -> str:
    return (
        "You are Launch Buddy, a Product Hunt assistant. Use the provided tools to look up real launch data "
        "before replying. Always cite the source of any metrics or posts you mention. When highlighting "
        "launches, include the name, tagline, vote count, and a link to the post. If a tool returns no data, "
        "explain that transparently and suggest alternative timeframes or search keywords."
    )


def create_product_hunt_crew(settings: ProductHuntSettings) -> Crew:
    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.base_url,
        temperature=0.6,
    )

    @tool("getTopProducts")
    def tool_get_top_products(limit: int = 3) -> Dict[str, Any]:
        safe_limit = _clamp(limit, 1, 10, 3)
        posts = asyncio.run(get_top_products_by_votes(safe_limit, settings))
        return {
            "posts": posts,
            "limit": safe_limit,
            "order": "VOTES",
        }

    @tool("getTopProductsThisWeek")
    def tool_get_top_products_this_week(limit: int = 3, days: int = 7) -> Dict[str, Any]:
        safe_limit = _clamp(limit, 1, 10, 3)
        safe_days = _clamp(days, 1, 31, 7)
        posts = asyncio.run(get_top_products_this_week(safe_limit, safe_days, settings))
        return {
            "posts": posts,
            "limit": safe_limit,
            "days": safe_days,
            "window": "rolling-week",
            "order": "RANKING",
        }

    @tool("getTopProductsByTimeframe")
    def tool_get_top_products_by_timeframe(
        timeframe: Optional[str] = None,
        tz: Optional[str] = None,
        limit: int = 3,
    ) -> Dict[str, Any]:
        safe_limit = _clamp(limit, 1, 10, 3)
        posts = asyncio.run(
            get_top_products_by_timeframe(
                first=safe_limit,
                timeframe=timeframe,
                tz=tz,
                settings=settings,
            )
        )
        window = parse_timeframe(timeframe, tz, settings=settings)
        return {
            "posts": posts,
            "limit": safe_limit,
            "timeframe": timeframe or "today",
            "tz": tz or settings.default_timezone,
            "window": window,
            "order": "RANKING",
        }

    @tool("searchProducts")
    def tool_search_products(query: str, limit: int = 10) -> Dict[str, Any]:
        safe_limit = _clamp(limit, 1, 50, 10)
        hits = asyncio.run(search_products(query, limit=safe_limit, settings=settings))
        return {
            "hits": hits,
            "query": query,
            "limit": safe_limit,
        }

    @tool("triggerConfetti")
    def tool_trigger_confetti(
        reason: Optional[str] = None,
        colors: Optional[List[str]] = None,
        particle_count: int = 200,
        spread: int = 90,
        start_velocity: int = 45,
        origin: Optional[Dict[str, float]] = None,
        shapes: Optional[List[str]] = None,
        ticks: int = 200,
        disable_sound: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "reason": reason,
            "colors": colors or ["#ff577f", "#ff884b", "#ffd384", "#fff9b0", "#00c2ff", "#7b5cff"],
            "particleCount": _clamp(particle_count, 20, 1000, 200),
            "spread": _clamp(spread, 1, 360, 90),
            "startVelocity": _clamp(start_velocity, 1, 200, 45),
            "origin": origin or {"x": 0.5, "y": 0.5},
            "shapes": shapes or ["square", "circle"],
            "ticks": _clamp(ticks, 10, 5000, 200),
            "disableSound": disable_sound,
        }
        return {"confetti": payload}

    agent = Agent(
        name="ProductHuntAgent",
        role="Launch Buddy",
        goal="Help founders and marketers explore Product Hunt launches with live data.",
        backstory="A pragmatic assistant that checks real Product Hunt endpoints before answering.",
        tools=[
            tool_get_top_products,
            tool_get_top_products_this_week,
            tool_get_top_products_by_timeframe,
            tool_search_products,
            tool_trigger_confetti,
        ],
        llm=model,
        allow_delegation=False,
        verbose=False,
    )

    task = Task(
        description=(
            "Use the available Product Hunt tools to gather data before answering.\n"
            "Conversation so far:\n{conversation}\n\nLatest question: {question}\n"
            "Include links and vote counts where possible. If data is missing, be transparent and suggest alternatives."
        ),
        expected_output="Helpful answer that cites Product Hunt data or clearly states when no data is available.",
        agent=agent,
    )

    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )


__all__ = ["create_product_hunt_crew", "build_system_prompt"]
