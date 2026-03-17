"""GitHub tools: issues, comments, reactions."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from ouroboros.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api(method: str, path: str, data: Optional[Dict[str, Any]] = None) -> Any:
    """Call the GitHub REST API and return parsed JSON."""
    token = os.environ.get("GITHUB_TOKEN", "")
    url = f"https://api.github.com{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            message = json.loads(e.read().decode()).get("message", e.reason)
        except Exception:
            message = e.reason
        return {"error": f"{e.code}: {message}"}
    except Exception as exc:
        return {"error": str(exc)}


def _get_repo_slug(ctx: ToolContext) -> str:
    """Get 'owner/repo' by parsing .git/config for the origin remote URL."""
    try:
        git_config = os.path.join(str(ctx.repo_dir), ".git", "config")
        with open(git_config, "r", encoding="utf-8") as f:
            content = f.read()
        # Find the [remote "origin"] section and extract its url
        m = re.search(r'\[remote "origin"\][^\[]*url\s*=\s*(\S+)', content, re.DOTALL)
        if m:
            remote_url = m.group(1)
            # Match both https (github.com/owner/repo) and ssh (github.com:owner/repo)
            slug_m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", remote_url)
            if slug_m:
                return slug_m.group(1)
    except Exception:
        log.debug("Failed to get repo slug from .git/config", exc_info=True)
    user = os.environ.get("GITHUB_USER", "")
    repo = os.environ.get("GITHUB_REPO", "")
    return f"{user}/{repo}"


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _list_issues(ctx: ToolContext, state: str = "open", labels: str = "", limit: int = 20) -> str:
    """List GitHub issues with optional filters."""
    slug = _get_repo_slug(ctx)
    per_page = min(limit, 50)
    path = f"/repos/{slug}/issues?state={state}&per_page={per_page}"
    if labels:
        path += f"&labels={urllib.parse.quote(labels)}"

    result = _api("GET", path)
    if isinstance(result, dict) and "error" in result:
        return f"⚠️ GH_ERROR: {result['error']}"
    if not isinstance(result, list):
        return f"⚠️ Unexpected response: {str(result)[:500]}"

    issues = result
    if not issues:
        return f"No {state} issues found."

    lines = [f"**{len(issues)} {state} issue(s):**\n"]
    for issue in issues:
        labels_str = ", ".join(lbl.get("name", "") for lbl in issue.get("labels", []))
        author = (issue.get("user") or {}).get("login", "unknown")
        suffix = (", labels: " + labels_str) if labels_str else ""
        lines.append(f"- **#{issue['number']}** {issue['title']} (by @{author}{suffix})")
        body_text = (issue.get("body") or "").strip()
        if body_text:
            preview = body_text[:200] + ("..." if len(body_text) > 200 else "")
            lines.append(f"  > {preview}")

    return "\n".join(lines)


def _get_issue(ctx: ToolContext, number: int) -> str:
    """Get a single issue with full details and comments."""
    if number <= 0:
        return "⚠️ issue number must be positive"

    slug = _get_repo_slug(ctx)

    issue = _api("GET", f"/repos/{slug}/issues/{number}")
    if isinstance(issue, dict) and "error" in issue:
        return f"⚠️ GH_ERROR: {issue['error']}"

    labels_str = ", ".join(lbl.get("name", "") for lbl in issue.get("labels", []))
    author = (issue.get("user") or {}).get("login", "unknown")

    lines = [
        f"## Issue #{issue['number']}: {issue['title']}",
        f"**State:** {issue['state']}  |  **Author:** @{author}",
    ]
    if labels_str:
        lines.append(f"**Labels:** {labels_str}")

    body_text = (issue.get("body") or "").strip()
    if body_text:
        lines.append(f"\n**Body:**\n{body_text[:3000]}")

    comments_data = _api("GET", f"/repos/{slug}/issues/{number}/comments")
    if isinstance(comments_data, list) and comments_data:
        lines.append(f"\n**Comments ({len(comments_data)}):**")
        for c in comments_data[:10]:
            c_author = (c.get("user") or {}).get("login", "unknown")
            c_body = (c.get("body") or "").strip()[:500]
            lines.append(f"\n@{c_author}:\n{c_body}")

    return "\n".join(lines)


def _comment_on_issue(ctx: ToolContext, number: int, body: str) -> str:
    """Add a comment to an issue."""
    if number <= 0:
        return "⚠️ issue number must be positive"
    if not body or not body.strip():
        return "⚠️ Comment body cannot be empty."

    slug = _get_repo_slug(ctx)
    result = _api("POST", f"/repos/{slug}/issues/{number}/comments", {"body": body})
    if isinstance(result, dict) and "error" in result:
        return f"⚠️ GH_ERROR: {result['error']}"
    return f"✅ Comment added to issue #{number}."


def _close_issue(ctx: ToolContext, number: int, comment: str = "") -> str:
    """Close an issue with optional closing comment."""
    if number <= 0:
        return "⚠️ issue number must be positive"

    if comment and comment.strip():
        res = _comment_on_issue(ctx, number, comment)
        if res.startswith("⚠️"):
            return res

    slug = _get_repo_slug(ctx)
    result = _api("PATCH", f"/repos/{slug}/issues/{number}", {"state": "closed"})
    if isinstance(result, dict) and "error" in result:
        return f"⚠️ GH_ERROR: {result['error']}"
    return f"✅ Issue #{number} closed."


def _create_issue(ctx: ToolContext, title: str, body: str = "", labels: str = "") -> str:
    """Create a new GitHub issue."""
    if not title or not title.strip():
        return "⚠️ Issue title cannot be empty."

    slug = _get_repo_slug(ctx)
    label_list = [lbl.strip() for lbl in labels.split(",") if lbl.strip()] if labels else []
    payload: Dict[str, Any] = {"title": title, "body": body, "labels": label_list}
    result = _api("POST", f"/repos/{slug}/issues", payload)
    if isinstance(result, dict) and "error" in result:
        return f"⚠️ GH_ERROR: {result['error']}"
    url = result.get("html_url") or ("#" + str(result.get("number", "?")))
    return f"✅ Issue created: {url}"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("list_github_issues", {
            "name": "list_github_issues",
            "description": "List GitHub issues. Use to check for new tasks, bug reports, or feature requests from the creator or contributors.",
            "parameters": {"type": "object", "properties": {
                "state": {"type": "string", "default": "open", "enum": ["open", "closed", "all"], "description": "Filter by state"},
                "labels": {"type": "string", "default": "", "description": "Filter by label (comma-separated)"},
                "limit": {"type": "integer", "default": 20, "description": "Max issues to return (max 50)"},
            }, "required": []},
        }, _list_issues),

        ToolEntry("get_github_issue", {
            "name": "get_github_issue",
            "description": "Get full details of a GitHub issue including body and comments.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "description": "Issue number"},
            }, "required": ["number"]},
        }, _get_issue),

        ToolEntry("comment_on_issue", {
            "name": "comment_on_issue",
            "description": "Add a comment to a GitHub issue. Use to respond to issues, share progress, or ask clarifying questions.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "description": "Issue number"},
                "body": {"type": "string", "description": "Comment text (markdown)"},
            }, "required": ["number", "body"]},
        }, _comment_on_issue),

        ToolEntry("close_github_issue", {
            "name": "close_github_issue",
            "description": "Close a GitHub issue with optional closing comment.",
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "description": "Issue number"},
                "comment": {"type": "string", "default": "", "description": "Optional closing comment"},
            }, "required": ["number"]},
        }, _close_issue),

        ToolEntry("create_github_issue", {
            "name": "create_github_issue",
            "description": "Create a new GitHub issue. Use for tracking tasks, documenting bugs, or planning features.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string", "description": "Issue title"},
                "body": {"type": "string", "default": "", "description": "Issue body (markdown)"},
                "labels": {"type": "string", "default": "", "description": "Labels (comma-separated)"},
            }, "required": ["title"]},
        }, _create_issue),
    ]
