"""Web search tool — multi-provider with caching and retry.

Provider strategy (in order):
  1. DuckDuckGo (ddgs) — free, no API key, no quota.
  2. Brave Search API — if BRAVE_SEARCH_API_KEY is set (1000 free/month).
  3. Google Custom Search — if GOOGLE_CSE_API_KEY + GOOGLE_CSE_ID are set (100 free/day).
  4. OpenAI Responses API — if OPENAI_API_KEY is set (fallback, has quota).

Additional features:
  - In-memory result cache (TTL 1 hour) to avoid duplicate queries.
  - Retry with exponential backoff for transient DDGS failures.
  - Structured JSON output: {"answer": "...", "sources": [...], "provider": "..."}
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.tools.registry import ToolContext, ToolEntry


# ---------------------------------------------------------------------------
# Simple in-memory cache with TTL
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL = 3600  # 1 hour


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _CACHE.get(key)
    if entry and (time.time() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: Dict[str, Any]) -> None:
    _CACHE[key] = (time.time(), value)


def _cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


# ---------------------------------------------------------------------------
# Provider 1: DuckDuckGo (free, no API key)
# ---------------------------------------------------------------------------

def _search_ddgs(query: str, max_results: int = 6, max_retries: int = 2) -> Dict[str, Any]:
    """Search via DuckDuckGo — free, no API key required. Retries on transient errors."""
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        return {"error": "ddgs package not installed; run: pip install ddgs h2"}

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            ddgs = DDGS()
            raw = list(ddgs.text(query, max_results=max_results))
            if not raw:
                return {"answer": "(no results found)", "sources": []}

            parts: List[str] = []
            sources: List[Dict[str, str]] = []
            for r in raw:
                title = r.get("title", "")
                body = r.get("body", "")
                url = r.get("href", "")
                parts.append(f"**{title}**\n{body}")
                if url:
                    sources.append({"title": title, "url": url})

            return {"answer": "\n\n".join(parts), "sources": sources}
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 ** attempt)  # backoff: 1s, 1.5s

    return {"error": f"DDGS error after {max_retries + 1} attempts: {last_error!r}"}


# ---------------------------------------------------------------------------
# Provider 2: Brave Search API (1000 free/month with API key)
# ---------------------------------------------------------------------------

def _search_brave(query: str, max_results: int = 6) -> Dict[str, Any]:
    """Search via Brave Search API. Requires BRAVE_SEARCH_API_KEY env var."""
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        return {"error": "BRAVE_SEARCH_API_KEY not set"}

    try:
        import urllib.request
        import urllib.parse

        params = urllib.parse.urlencode({"q": query, "count": max_results, "search_lang": "en"})
        url = f"https://api.search.brave.com/res/v1/web/search?{params}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        web_results = data.get("web", {}).get("results", [])
        if not web_results:
            return {"answer": "(no results found)", "sources": []}

        parts: List[str] = []
        sources: List[Dict[str, str]] = []
        for r in web_results:
            title = r.get("title", "")
            description = r.get("description", "")
            url_r = r.get("url", "")
            parts.append(f"**{title}**\n{description}")
            if url_r:
                sources.append({"title": title, "url": url_r})

        return {"answer": "\n\n".join(parts), "sources": sources}
    except Exception as exc:
        return {"error": f"Brave Search error: {exc!r}"}


# ---------------------------------------------------------------------------
# Provider 3: Google Custom Search (100 free/day with API key)
# ---------------------------------------------------------------------------

def _search_google_cse(query: str, max_results: int = 6) -> Dict[str, Any]:
    """Search via Google Custom Search Engine. Requires GOOGLE_CSE_API_KEY + GOOGLE_CSE_ID."""
    api_key = os.environ.get("GOOGLE_CSE_API_KEY", "")
    cse_id = os.environ.get("GOOGLE_CSE_ID", "")
    if not api_key or not cse_id:
        return {"error": "GOOGLE_CSE_API_KEY and/or GOOGLE_CSE_ID not set"}

    try:
        import urllib.request
        import urllib.parse

        params = urllib.parse.urlencode({
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": min(max_results, 10),
        })
        url = f"https://www.googleapis.com/customsearch/v1?{params}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())

        items = data.get("items", [])
        if not items:
            return {"answer": "(no results found)", "sources": []}

        parts: List[str] = []
        sources: List[Dict[str, str]] = []
        for r in items:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            parts.append(f"**{title}**\n{snippet}")
            if link:
                sources.append({"title": title, "url": link})

        return {"answer": "\n\n".join(parts), "sources": sources}
    except Exception as exc:
        return {"error": f"Google CSE error: {exc!r}"}


# ---------------------------------------------------------------------------
# Provider 4: OpenAI Responses API (fallback, has quota limits)
# ---------------------------------------------------------------------------

def _search_openai(query: str) -> Dict[str, Any]:
    """Search via OpenAI Responses API (requires quota)."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        model = os.environ.get("OUROBOROS_WEBSEARCH_MODEL", "gpt-4o-mini")
        # Strip provider prefix if present
        if "/" in model:
            model = model.split("/", 1)[1]

        resp = client.responses.create(
            model=model,
            tools=[{"type": "web_search_preview"}],
            tool_choice="auto",
            input=query,
        )
        d = resp.model_dump()
        text = ""
        for item in d.get("output", []) or []:
            if item.get("type") == "message":
                for block in item.get("content", []) or []:
                    if block.get("type") in ("output_text", "text"):
                        text += block.get("text", "")
        return {"answer": text or "(no answer)", "sources": []}
    except Exception as exc:
        return {"error": f"OpenAI error: {exc!r}"}


# ---------------------------------------------------------------------------
# Main tool handler
# ---------------------------------------------------------------------------

def _web_search(ctx: ToolContext, query: str) -> str:
    """Multi-provider web search with caching. Tries providers in order until one succeeds."""

    # Check cache first
    cache_key = _cache_key(query)
    cached = _cache_get(cache_key)
    if cached:
        cached["cached"] = True
        return json.dumps(cached, ensure_ascii=False, indent=2)

    errors: Dict[str, str] = {}

    # 1. Try DDGS (free, no quota)
    result = _search_ddgs(query)
    if "error" not in result:
        result["provider"] = "duckduckgo"
        _cache_set(cache_key, result)
        return json.dumps(result, ensure_ascii=False, indent=2)
    errors["duckduckgo"] = result["error"]

    # 2. Try Brave Search (if API key configured)
    if os.environ.get("BRAVE_SEARCH_API_KEY"):
        result = _search_brave(query)
        if "error" not in result:
            result["provider"] = "brave"
            _cache_set(cache_key, result)
            return json.dumps(result, ensure_ascii=False, indent=2)
        errors["brave"] = result["error"]

    # 3. Try Google Custom Search (if keys configured)
    if os.environ.get("GOOGLE_CSE_API_KEY") and os.environ.get("GOOGLE_CSE_ID"):
        result = _search_google_cse(query)
        if "error" not in result:
            result["provider"] = "google_cse"
            _cache_set(cache_key, result)
            return json.dumps(result, ensure_ascii=False, indent=2)
        errors["google_cse"] = result["error"]

    # 4. Fallback: OpenAI (has quota limits)
    result = _search_openai(query)
    if "error" not in result:
        result["provider"] = "openai"
        _cache_set(cache_key, result)
        return json.dumps(result, ensure_ascii=False, indent=2)
    errors["openai"] = result["error"]

    # All providers failed
    return json.dumps({
        "error": "All search providers failed",
        "provider_errors": errors,
        "hint": (
            "DDGS may be rate-limited (transient — try again in a few minutes). "
            "To add more providers: set BRAVE_SEARCH_API_KEY (1000 free/month at https://brave.com/search/api/) "
            "or GOOGLE_CSE_API_KEY + GOOGLE_CSE_ID (100 free/day at https://programmablesearchengine.google.com/). "
            "OpenAI quota: check https://platform.openai.com/usage"
        ),
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            "web_search",
            {
                "name": "web_search",
                "description": (
                    "Search the web via DuckDuckGo (primary, free) with OpenAI Responses API as fallback. "
                    "Returns JSON with answer + sources."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
            _web_search,
        ),
    ]
