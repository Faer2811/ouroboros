"""Web search tool — DuckDuckGo primary, OpenAI Responses API fallback.

Strategy:
  1. Try DuckDuckGo (ddgs) — free, no API key, no quota.
  2. If DDGS fails and OPENAI_API_KEY is set → try OpenAI Responses API.
  3. Return structured JSON: {"answer": "...", "sources": [...]}

Installing ddgs happens lazily; the package is installed in colab_launcher.py
or falls back gracefully if unavailable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _search_ddgs(query: str, max_results: int = 6) -> Dict[str, Any]:
    """Search via DuckDuckGo — free, no API key required."""
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        return {"error": "ddgs package not installed; run: pip install ddgs"}

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
        return {"error": f"DDGS error: {exc!r}"}


def _search_openai(query: str) -> Dict[str, Any]:
    """Search via OpenAI Responses API (requires quota)."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        model = os.environ.get("OUROBOROS_WEBSEARCH_MODEL", "gpt-4o-mini")
        if model == "openai/gpt-4o":
            model = "gpt-4o-mini"

        resp = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
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
    """Primary: DuckDuckGo. Fallback: OpenAI Responses API."""

    # 1. Try DDGS (free, no quota)
    result = _search_ddgs(query)
    if "error" not in result:
        result["provider"] = "duckduckgo"
        return json.dumps(result, ensure_ascii=False, indent=2)

    ddgs_error = result["error"]

    # 2. Fallback: OpenAI
    result = _search_openai(query)
    if "error" not in result:
        result["provider"] = "openai"
        return json.dumps(result, ensure_ascii=False, indent=2)

    # 3. Both failed — return combined error
    return json.dumps({
        "error": "All search providers failed",
        "ddgs_error": ddgs_error,
        "openai_error": result["error"],
        "hint": (
            "DDGS failed (network/rate-limit) and OpenAI quota may be exhausted. "
            "Try again later or check OpenAI billing at https://platform.openai.com/usage"
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
                    "Search the web via DuckDuckGo (primary, free) with OpenAI "
                    "Responses API as fallback. Returns JSON with answer + sources."
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
