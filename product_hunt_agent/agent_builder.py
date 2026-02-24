from __future__ import annotations

import asyncio
from typing import Any, Dict

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

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
        "You are Launch Buddy, a Product Hunt rankings analyst and launch strategist. "
        "Use the provided tools to look up real launch data before replying. "
        "When you mention rankings or metrics, cite Product Hunt as the source and include "
        "the product name, tagline, vote count, and link. "
        "If a tool returns no data, explain why and suggest a different timeframe or search query. "
        "Never invent metrics or posts."
    )


def create_product_hunt_crew(settings: ProductHuntSettings, *, stream: bool = False) -> Crew:
    model = LLM(model=settings.openai_model, api_key=settings.openai_api_key, base_url=settings.base_url, temperature=0.6)

    @tool("getTopProducts")
    def tool_get_top_products(limit: int = 3) -> Dict[str, Any]:
        """Get top Product Hunt products sorted by vote count. Use this for all-time top products."""
        safe_limit = _clamp(limit, 1, 10, 3)
        posts = asyncio.run(get_top_products_by_votes(safe_limit, settings))
        return {
            "posts": posts,
            "limit": safe_limit,
            "order": "VOTES",
        }

    @tool("getTopProductsThisWeek")
    def tool_get_top_products_this_week(limit: int = 3, days: int = 7) -> Dict[str, Any]:
        """Get top Product Hunt products from a rolling time window. Use for recent launches like 'this week' or 'last few days'."""
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
        timeframe: str | None = None,
        tz: str | None = None,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """Get top Product Hunt products for a specific timeframe like 'today', 'yesterday', 'last_week', or 'last_month'. Use limit=1 for a single top result."""
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
        """Search Product Hunt for products matching a keyword or phrase."""
        safe_limit = _clamp(limit, 1, 50, 10)
        hits = asyncio.run(search_products(query, limit=safe_limit, settings=settings))
        return {
            "hits": hits,
            "query": query,
            "limit": safe_limit,
        }

    @tool("triggerConfetti")
    def tool_trigger_confetti(
        reason: str | None = None,
        colors: list[str] | None = None,
        particle_count: int = 200,
        spread: int = 90,
        start_velocity: int = 45,
        origin: dict[str, float] | None = None,
        shapes: list[str] | None = None,
        ticks: int = 200,
        disable_sound: bool = True,
    ) -> Dict[str, Any]:
        """Trigger a confetti celebration animation. Use to celebrate achievements or milestones."""
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
        role="Product Hunt rankings analyst and launch strategist for early-stage SaaS/AI products",
        goal=(
            "Deliver accurate, actionable answers grounded in live Product Hunt data. "
            "When asked for rankings, return the correct top results with vote counts and links. "
            "When asked for launch advice, provide concise, practical guidance."
        ),
        backstory=(
            "You analyze Product Hunt leaderboards daily and advise founders on launch strategy. "
            "You value evidence over hype, rely on tools for data, and are transparent when data is missing."
        ),
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
        memory=False,
    )

    task = Task(
        description=(
            "Answer the user's Product Hunt question using available tools when data is required.\n"
            "Inputs:\n"
            "- Conversation: {conversation}\n"
            "- Latest question: {question}\n"
            "- Intent hint (may be empty): {intent_hint}\n"
            "Process:\n"
            "1. Decide if the user needs rankings, search results, or launch advice.\n"
            "2. For rankings, call the correct tool and apply any timeframe or limit hints.\n"
            "3. If the question asks for a single top result (highest/#1), use limit=1.\n"
            "4. Use tool output verbatim for names, taglines, votes, and links; do not invent data.\n"
            "5. If data is missing (e.g., no API token), explain and suggest alternatives.\n"
            "6. Do not reveal internal reasoning or tool execution steps.\n"
        ),
        expected_output=(
            "Markdown response with a single, clear purpose.\n"
            "If rankings: 1-sentence summary + bullet list of results (Name — Tagline — Votes — Link) + timeframe notes.\n"
            "If search: 1-sentence summary + bullet list of results (Name — Tagline — Votes — Link).\n"
            "If advice: 3-6 bullets of actionable guidance.\n"
            "Always mention Product Hunt as the data source when metrics are included."
        ),
        agent=agent,
    )

    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        stream=stream,
    )


__all__ = ["create_product_hunt_crew", "build_system_prompt"]
