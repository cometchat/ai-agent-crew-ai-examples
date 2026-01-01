from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import pendulum

from .config import ProductHuntSettings

PRODUCT_HUNT_ENDPOINT = "https://api.producthunt.com/v2/api/graphql"
DEFAULT_APP_INDEX = "Post_production"


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def has_producthunt_token(settings: ProductHuntSettings) -> bool:
    return bool(settings.producthunt_api_token and settings.producthunt_api_token.strip())


def parse_timeframe(
    timeframe: Optional[str],
    tz: Optional[str],
    *,
    settings: ProductHuntSettings,
    now: Optional[datetime] = None,
) -> Dict[str, str]:
    zone_name = tz or settings.default_timezone
    try:
        zone = pendulum.timezone(zone_name)
    except pendulum.tz.zoneinfo.exceptions.InvalidTimezone:
        zone = pendulum.timezone(settings.default_timezone)

    baseline = pendulum.instance(now or datetime.utcnow(), tz=zone)
    normalized = (timeframe or "today").strip().lower()

    date_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", normalized)
    if date_match:
        start = zone.datetime(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)), 0, 0, 0)
        end = start.add(days=1)
        return {
            "postedAfter": start.in_timezone("UTC").to_iso8601_string(),
            "postedBefore": end.in_timezone("UTC").to_iso8601_string(),
            "label": "day",
        }

    range_match = re.search(r"from[:=]\s*(\d{4}-\d{2}-\d{2}).*to[:=]\s*(\d{4}-\d{2}-\d{2})", normalized)
    if range_match:
        start = zone.datetime(*map(int, range_match.group(1).split("-")), 0, 0, 0)
        end = zone.datetime(*map(int, range_match.group(2).split("-")), 0, 0, 0).add(days=1)
        return {
            "postedAfter": start.in_timezone("UTC").to_iso8601_string(),
            "postedBefore": end.in_timezone("UTC").to_iso8601_string(),
            "label": "range",
        }

    if not normalized or "today" in normalized:
        start = baseline.start_of("day")
        end = start.add(days=1)
        return {
            "postedAfter": start.in_timezone("UTC").to_iso8601_string(),
            "postedBefore": end.in_timezone("UTC").to_iso8601_string(),
            "label": "today",
        }

    if "yesterday" in normalized:
        end = baseline.start_of("day")
        start = end.subtract(days=1)
        return {
            "postedAfter": start.in_timezone("UTC").to_iso8601_string(),
            "postedBefore": end.in_timezone("UTC").to_iso8601_string(),
            "label": "yesterday",
        }

    if "last week" in normalized or "last-week" in normalized:
        end = baseline.start_of("week")
        start = end.subtract(weeks=1)
        return {
            "postedAfter": start.in_timezone("UTC").to_iso8601_string(),
            "postedBefore": end.in_timezone("UTC").to_iso8601_string(),
            "label": "last-week",
        }

    if "this week" in normalized or "this-week" in normalized or "week" in normalized:
        start = baseline.start_of("week")
        end = baseline
        return {
            "postedAfter": start.in_timezone("UTC").to_iso8601_string(),
            "postedBefore": end.in_timezone("UTC").to_iso8601_string(),
            "label": "this-week",
        }

    if "last month" in normalized or "last-month" in normalized:
        end = baseline.start_of("month")
        start = end.subtract(months=1)
        return {
            "postedAfter": start.in_timezone("UTC").to_iso8601_string(),
            "postedBefore": end.in_timezone("UTC").to_iso8601_string(),
            "label": "last-month",
        }

    if "this month" in normalized or "this-month" in normalized or "month" in normalized:
        start = baseline.start_of("month")
        end = baseline
        return {
            "postedAfter": start.in_timezone("UTC").to_iso8601_string(),
            "postedBefore": end.in_timezone("UTC").to_iso8601_string(),
            "label": "this-month",
        }

    rolling_days = re.search(r"(?:past|last)\s+(\d{1,2})\s+day", normalized)
    if rolling_days:
        count = clamp(int(rolling_days.group(1)), 1, 31)
        start = baseline.subtract(days=count)
        return {
            "postedAfter": start.in_timezone("UTC").to_iso8601_string(),
            "postedBefore": baseline.in_timezone("UTC").to_iso8601_string(),
            "label": f"last-{count}-days",
        }

    start = baseline.start_of("day")
    end = start.add(days=1)
    return {
        "postedAfter": start.in_timezone("UTC").to_iso8601_string(),
        "postedBefore": end.in_timezone("UTC").to_iso8601_string(),
        "label": "today",
    }


async def fetch_graphql(
    query: str,
    *,
    settings: ProductHuntSettings,
    variables: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not has_producthunt_token(settings):
        return None

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {settings.producthunt_api_token}",
    }
    payload: Dict[str, Any] = {"query": query}
    if variables is not None:
        payload["variables"] = variables

    timeout = httpx.Timeout(settings.http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(PRODUCT_HUNT_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("data")
        except (httpx.HTTPError, json.JSONDecodeError):
            return None


def map_edges_to_posts(edges: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    posts: List[Dict[str, Any]] = []
    if not edges:
        return posts
    for edge in edges:
        node = edge.get("node") if isinstance(edge, dict) else None
        if not node:
            continue
        posts.append(
            {
                "id": node.get("id"),
                "name": node.get("name"),
                "tagline": node.get("tagline"),
                "url": node.get("url"),
                "votesCount": node.get("votesCount"),
            }
        )
    return posts


async def get_top_products_by_votes(first: int, settings: ProductHuntSettings) -> List[Dict[str, Any]]:
    if not has_producthunt_token(settings):
        return []
    safe_first = clamp(first, 1, 50)
    query = """
    query TopByVotes($first: Int!) {
      posts(order: VOTES, first: $first) {
        edges {
          node { id name tagline url votesCount }
        }
      }
    }
    """
    data = await fetch_graphql(query, settings=settings, variables={"first": safe_first})
    edges = data.get("posts", {}).get("edges") if data else []
    return map_edges_to_posts(edges)


async def get_top_products_this_week(first: int, days: int, settings: ProductHuntSettings) -> List[Dict[str, Any]]:
    if not has_producthunt_token(settings):
        return []
    safe_first = clamp(first, 1, 50)
    safe_days = clamp(days, 1, 31)

    now = datetime.utcnow()
    posted_before = now.isoformat() + "Z"
    posted_after = (now - timedelta(days=safe_days)).isoformat() + "Z"

    query = """
    query TopWeek($first: Int!, $postedAfter: DateTime!, $postedBefore: DateTime!) {
      posts(first: $first, order: RANKING, postedAfter: $postedAfter, postedBefore: $postedBefore) {
        edges { node { id name tagline url votesCount } }
      }
    }
    """
    variables = {
        "first": safe_first,
        "postedAfter": posted_after,
        "postedBefore": posted_before,
    }
    data = await fetch_graphql(query, settings=settings, variables=variables)
    edges = data.get("posts", {}).get("edges") if data else []
    return map_edges_to_posts(edges)


async def get_top_products_by_timeframe(
    *,
    first: int,
    timeframe: Optional[str],
    tz: Optional[str],
    settings: ProductHuntSettings,
) -> List[Dict[str, Any]]:
    if not has_producthunt_token(settings):
        return []
    safe_first = clamp(first, 1, 50)
    window = parse_timeframe(timeframe, tz, settings=settings)
    query = """
    query TopByTimeframe($first: Int!, $postedAfter: DateTime!, $postedBefore: DateTime!) {
      posts(first: $first, order: RANKING, postedAfter: $postedAfter, postedBefore: $postedBefore) {
        edges { node { id name tagline url votesCount } }
      }
    }
    """
    variables = {
        "first": safe_first,
        "postedAfter": window["postedAfter"],
        "postedBefore": window["postedBefore"],
    }
    data = await fetch_graphql(query, settings=settings, variables=variables)
    edges = data.get("posts", {}).get("edges") if data else []
    if edges:
        return map_edges_to_posts(edges)

    inline_query = f"""
    query {{
      posts(first: {safe_first}, order: RANKING, postedAfter: "{window['postedAfter']}", postedBefore: "{window['postedBefore']}") {{
        edges {{ node {{ id name tagline url votesCount }} }}
      }}
    }}
    """
    inline_data = await fetch_graphql(inline_query, settings=settings)
    inline_edges = inline_data.get("posts", {}).get("edges") if inline_data else []
    return map_edges_to_posts(inline_edges)


async def search_products(query: str, *, limit: int, settings: ProductHuntSettings) -> List[Dict[str, Any]]:
    safe_limit = clamp(limit, 1, 50)
    base_url = f"https://{settings.algolia_app_id.lower()}-dsn.algolia.net/1/indexes/{DEFAULT_APP_INDEX}"
    headers = {
        "X-Algolia-Application-Id": settings.algolia_app_id,
        "X-Algolia-API-Key": settings.algolia_api_key,
    }
    timeout = httpx.Timeout(settings.http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(
                base_url,
                headers=headers,
                params={"query": query, "hitsPerPage": safe_limit},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return []

    hits = payload.get("hits") if isinstance(payload, dict) else None
    results: List[Dict[str, Any]] = []
    if isinstance(hits, list):
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            results.append(
                {
                    "objectID": hit.get("objectID") or hit.get("id"),
                    "name": hit.get("name"),
                    "tagline": hit.get("tagline") or hit.get("tag_line") or hit.get("tagLine"),
                    "url": hit.get("url") or hit.get("post_url"),
                    "votesCount": hit.get("votesCount") or hit.get("votes_count"),
                }
            )
    return results


__all__ = [
    "DEFAULT_APP_INDEX",
    "PRODUCT_HUNT_ENDPOINT",
    "get_top_products_by_timeframe",
    "get_top_products_by_votes",
    "get_top_products_this_week",
    "has_producthunt_token",
    "parse_timeframe",
    "search_products",
]
