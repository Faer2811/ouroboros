"""
Supervisor — Telegram client + formatting.

TelegramClient, message splitting, markdown→HTML conversion, send_with_budget.
"""

from __future__ import annotations

import datetime
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from supervisor.state import load_state, save_state, append_jsonl, \
    get_user_session, start_user_session, update_user_session, end_user_session
from supervisor.portrait import check_portrait_trigger

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level config (set via init())
# ---------------------------------------------------------------------------
DRIVE_ROOT = None  # pathlib.Path
TOTAL_BUDGET_LIMIT: float = 0.0
BUDGET_REPORT_EVERY_MESSAGES: int = 10
_TG: Optional["TelegramClient"] = None


def init(drive_root, total_budget_limit: float, budget_report_every: int,
         tg_client: "TelegramClient") -> None:
    global DRIVE_ROOT, TOTAL_BUDGET_LIMIT, BUDGET_REPORT_EVERY_MESSAGES, _TG
    DRIVE_ROOT = drive_root
    TOTAL_BUDGET_LIMIT = total_budget_limit
    BUDGET_REPORT_EVERY_MESSAGES = budget_report_every
    _TG = tg_client


def get_tg() -> "TelegramClient":
    assert _TG is not None, "telegram.init() not called"
    return _TG


# ---------------------------------------------------------------------------
# TelegramClient
# ---------------------------------------------------------------------------

class TelegramClient:
    def __init__(self, token: str):
        self.base = f"https://api.telegram.org/bot{token}"
        self._token = token

    def get_updates(self, offset: int, timeout: int = 10) -> List[Dict[str, Any]]:
        last_err = "unknown"
        for attempt in range(3):
            try:
                r = requests.get(
                    f"{self.base}/getUpdates",
                    params={"offset": offset, "timeout": timeout,
                            "allowed_updates": ["message", "edited_message"]},
                    timeout=timeout + 5,
                )
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is not True:
                    raise RuntimeError(f"Telegram getUpdates failed: {data}")
                return data.get("result") or []
            except Exception as e:
                last_err = repr(e)
                if attempt < 2:
                    import time
                    time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"Telegram getUpdates failed after retries: {last_err}")

    def send_message(self, chat_id: int, text: str, parse_mode: str = "") -> Tuple[bool, str]:
        last_err = "unknown"
        for attempt in range(3):
            try:
                payload: Dict[str, Any] = {"chat_id": chat_id, "text": text,
                                           "disable_web_page_preview": True}
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                r = requests.post(f"{self.base}/sendMessage", data=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is True:
                    return True, "ok"
                last_err = f"telegram_api_error: {data}"
            except Exception as e:
                last_err = repr(e)
            if attempt < 2:
                import time
                time.sleep(0.8 * (attempt + 1))
        return False, last_err

    def send_chat_action(self, chat_id: int, action: str = "typing") -> bool:
        """Send chat action (typing indicator). Best-effort, no retries."""
        try:
            r = requests.post(
                f"{self.base}/sendChatAction",
                data={"chat_id": chat_id, "action": action},
                timeout=5,
            )
            return r.status_code == 200
        except Exception:
            log.debug("Failed to send chat action to chat_id=%d", chat_id, exc_info=True)
            return False

    def send_photo(self, chat_id: int, photo_bytes: bytes,
                   caption: str = "") -> Tuple[bool, str]:
        """Send a photo to a chat. photo_bytes is raw PNG/JPEG data."""
        last_err = "unknown"
        for attempt in range(3):
            try:
                files = {"photo": ("screenshot.png", photo_bytes, "image/png")}
                data: Dict[str, Any] = {"chat_id": chat_id}
                if caption:
                    data["caption"] = caption[:1024]
                r = requests.post(
                    f"{self.base}/sendPhoto",
                    data=data, files=files, timeout=30,
                )
                r.raise_for_status()
                resp = r.json()
                if resp.get("ok") is True:
                    return True, "ok"
                last_err = f"telegram_api_error: {resp}"
            except Exception as e:
                last_err = repr(e)
            if attempt < 2:
                import time
                time.sleep(0.8 * (attempt + 1))
        return False, last_err

    def download_file_base64(self, file_id: str, max_bytes: int = 10_000_000) -> Tuple[Optional[str], str]:
        """Download a file from Telegram and return (base64_data, mime_type). Returns (None, "") on failure."""
        try:
            # Get file path
            r = requests.get(f"{self.base}/getFile", params={"file_id": file_id}, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                return None, ""
            file_path = data["result"].get("file_path", "")
            file_size = int(data["result"].get("file_size") or 0)
            if file_size > max_bytes:
                return None, ""

            # Download file
            download_url = f"https://api.telegram.org/file/bot{self._token}/{file_path}"
            r2 = requests.get(download_url, timeout=30)
            r2.raise_for_status()

            import base64
            b64 = base64.b64encode(r2.content).decode("ascii")

            # Guess mime type from extension
            ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                        "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp"}
            mime = mime_map.get(ext, "image/jpeg")  # default to jpeg

            return b64, mime
        except Exception:
            log.warning("Failed to download file_id=%s from Telegram", file_id, exc_info=True)
            return None, ""


# ---------------------------------------------------------------------------
# Message splitting + formatting
# ---------------------------------------------------------------------------

def split_telegram(text: str, limit: int = 3800) -> List[str]:
    chunks: List[str] = []
    s = text
    while len(s) > limit:
        cut = s.rfind("\n", 0, limit)
        if cut < 100:
            cut = limit
        chunks.append(s[:cut])
        s = s[cut:]
    chunks.append(s)
    return chunks


def _sanitize_telegram_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        c for c in text
        if (ord(c) >= 32 or c in ("\n", "\t")) and not (0xD800 <= ord(c) <= 0xDFFF)
    )


def _tg_utf16_len(text: str) -> int:
    if not text:
        return 0
    return sum(2 if ord(c) > 0xFFFF else 1 for c in text)


def _strip_markdown(text: str) -> str:
    """Strip all markdown formatting markers, leaving only plain text."""
    # Fenced code blocks (keep content)
    text = re.sub(r"```[^\n]*\n([\s\S]*?)```", r"\1", text)
    # Inline code (keep content)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold+italic (***text***)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
    # Bold (**text**)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # Italic (*text* or _text_)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    # Strikethrough (~~text~~)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # Links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Headers (# text -> text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # List markers (- or * at start of line, keep bullet but remove markdown)
    text = re.sub(r"^[\*\-]\s+", "• ", text, flags=re.MULTILINE)
    # Clean up any remaining stray markdown markers
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    text = text.replace("`", "")
    return text


def _markdown_to_telegram_html(md: str) -> str:
    """Convert Markdown to Telegram-safe HTML.

    Supported: fenced code, inline code, **bold**, *italic*, _italic_,
    ~~strikethrough~~, [links](url), # headers, list items.
    Handles unmatched markers gracefully. Telegram only allows: b, i, u, s, code, pre, a.
    """
    import html as _html
    md = md or ""

    # --- Step 1: extract fenced code blocks into placeholders ---
    # Match ``` with optional language, then content, then closing ```
    fence_re = re.compile(r"```[^\n]*\n([\s\S]*?)```", re.MULTILINE)
    fenced: list = []

    def _save_fence(m: re.Match) -> str:
        code_content = m.group(1)
        # Remove trailing newline if present
        if code_content.endswith("\n"):
            code_content = code_content[:-1]
        code_esc = _html.escape(code_content, quote=False)
        placeholder = f"\x00FENCE{len(fenced)}\x00"
        fenced.append(f"<pre>{code_esc}</pre>")
        return placeholder

    text = fence_re.sub(_save_fence, md)

    # --- Step 2: extract inline code into placeholders ---
    inline_code_re = re.compile(r"`([^`\n]+)`")
    inlines: list = []

    def _save_inline(m: re.Match) -> str:
        code_esc = _html.escape(m.group(1), quote=False)
        placeholder = f"\x00CODE{len(inlines)}\x00"
        inlines.append(f"<code>{code_esc}</code>")
        return placeholder

    text = inline_code_re.sub(_save_inline, text)

    # --- Step 3: HTML-escape remaining text (before adding HTML tags) ---
    text = _html.escape(text, quote=False)

    # --- Step 4: apply markdown formatting (order matters) ---
    # Headers: # at start of line -> bold with newline
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # Links: [text](url) - escape the URL too
    def _replace_link(m: re.Match) -> str:
        link_text = m.group(1)
        url = m.group(2)
        # URL must not contain quotes or special chars that break HTML
        url_safe = url.replace('"', '%22').replace('<', '%3C').replace('>', '%3E')
        return f'<a href="{url_safe}">{link_text}</a>'

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _replace_link, text)

    # Bold+italic: ***text*** (must come before ** and *)
    # Use non-greedy match, handle line breaks
    text = re.sub(r"\*\*\*([^*\n]+?)\*\*\*", r"<b><i>\1</i></b>", text)

    # Bold: **text** (non-greedy, single line)
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"<b>\1</b>", text)

    # Strikethrough: ~~text~~ (non-greedy, single line)
    text = re.sub(r"~~([^~\n]+?)~~", r"<s>\1</s>", text)

    # Italic: *text* (single *, not adjacent to another *, single line)
    # Lookahead/lookbehind to avoid matching ** or *** remnants
    text = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])", r"<i>\1</i>", text)

    # Italic: _text_ (word-boundary to avoid matching snake_case, single line)
    text = re.sub(r"\b_([^_\n]+?)_\b", r"<i>\1</i>", text)

    # List items: convert - or * at line start to •
    text = re.sub(r"^[\*\-]\s+", "• ", text, flags=re.MULTILINE)

    # --- Step 5: restore placeholders ---
    for i, code in enumerate(inlines):
        text = text.replace(f"\x00CODE{i}\x00", code)
    for i, block in enumerate(fenced):
        text = text.replace(f"\x00FENCE{i}\x00", block)

    return text


def _chunk_markdown_for_telegram(md: str, max_chars: int = 3500) -> List[str]:
    md = md or ""
    max_chars = max(256, min(4096, int(max_chars)))
    lines = md.splitlines(keepends=True)
    chunks: List[str] = []
    cur = ""
    in_fence = False
    fence_open = "```\n"
    fence_close = "```\n"

    def _flush() -> None:
        nonlocal cur
        if cur and cur.strip():
            chunks.append(cur)
        cur = ""

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            if in_fence:
                fence_open = line if line.endswith("\n") else (line + "\n")

        reserve = _tg_utf16_len(fence_close) if in_fence else 0
        if _tg_utf16_len(cur) + _tg_utf16_len(line) > max_chars - reserve:
            if in_fence and cur:
                cur += fence_close
            _flush()
            cur = fence_open if in_fence else ""
        cur += line

    if in_fence:
        cur += fence_close
    _flush()
    return chunks or [md]


def _send_markdown_telegram(chat_id: int, text: str) -> Tuple[bool, str]:
    """Send markdown text as Telegram HTML, with plain-text fallback."""
    tg = get_tg()
    chunks = _chunk_markdown_for_telegram(text or "", max_chars=3200)
    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
    if not chunks:
        return False, "empty_chunks"
    last_err = "ok"
    for md_part in chunks:
        html_text = _markdown_to_telegram_html(md_part)
        ok, err = tg.send_message(chat_id, _sanitize_telegram_text(html_text), parse_mode="HTML")
        if not ok:
            plain = _strip_markdown(md_part)
            if not plain.strip():
                return False, err
            ok2, err2 = tg.send_message(chat_id, _sanitize_telegram_text(plain))
            if not ok2:
                return False, err2
        last_err = err
    return True, last_err


# ---------------------------------------------------------------------------
# Budget + logging
# ---------------------------------------------------------------------------

def _format_budget_line(st: Dict[str, Any]) -> str:
    spent = float(st.get("spent_usd") or 0.0)
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    pct = (spent / total * 100.0) if total > 0 else 0.0
    sha = (st.get("current_sha") or "")[:8]
    branch = st.get("current_branch") or "?"
    return f"—\nBudget: ${spent:.4f} / ${total:.2f} ({pct:.2f}%) | {branch}@{sha}"


def budget_line(force: bool = False) -> str:
    """Budget line is disabled per creator request."""
    return ""


def log_supervisor(payload: Dict[str, Any]) -> None:
    append_jsonl(DRIVE_ROOT / "logs" / "supervisor.jsonl", {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **payload,
    })


def log_chat(direction: str, chat_id: int, user_id: int, text: str) -> None:
    append_jsonl(DRIVE_ROOT / "logs" / f"chat_{user_id}.jsonl", {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "direction": direction,
        "chat_id": chat_id,
        "user_id": user_id,
        "text": text[:800],
    })


# ---------------------------------------------------------------------------
# Public — send with markdown support
# ---------------------------------------------------------------------------

def send_with_budget(chat_id: int, text: str) -> None:
    _send_markdown_telegram(chat_id, text + budget_line())


# ---------------------------------------------------------------------------
# Startup greetings
# ---------------------------------------------------------------------------

def send_startup_greetings() -> None:
    """Send a greeting message to all users who have previously interacted with the agent.
    Only sends if greeting_sent_at is None or more than 6 hours ago.
    """
    from datetime import datetime, timezone, timedelta
    st = load_state()
    
    # Check last greeting time
    greeting_sent_at = st.get("greeting_sent_at")
    if greeting_sent_at:
        last_greeting = datetime.fromisoformat(greeting_sent_at)
        now = datetime.now(timezone.utc)
        if now - last_greeting < timedelta(hours=6):
            log.debug("Startup greetings skipped — last sent %s ago", now - last_greeting)
            return
    
    # Get unique user IDs: owner + allowed_users
    owner_id = st.get("owner_id", 0)
    allowed = st.get("allowed_user_ids", [])
    user_ids = list(set([owner_id] + allowed))
    user_ids = [uid for uid in user_ids if uid > 0]
    
    if not user_ids:
        log.debug("No user IDs to send startup greetings")
        return
    
    # Choose random greeting
    greetings = [
        "Привет! Я снова здесь 👋",
        "Доброе утро — чем могу помочь?",
        "Привет, не забыл про тебя. Есть вопросы?",
    ]
    msg = random.choice(greetings)
    
    # Send to each user
    tg = get_tg()
    for uid in user_ids:
        try:
            ok, err = tg.send_message(uid, msg)
            if not ok:
                log.warning("Failed to send startup greeting to user_id=%d: %s", uid, err)
            else:
                log.info("Sent startup greeting to user_id=%d", uid)
        except Exception as e:
            log.error("Error sending startup greeting to user_id=%d: %r", uid, e)
    
    # Update state
    st["greeting_sent_at"] = datetime.now(timezone.utc).isoformat()
    save_state(st)
    log.info("Startup greetings sent to %d users", len(user_ids))


# ---------------------------------------------------------------------------
# Incoming message handler
# ---------------------------------------------------------------------------

def handle_incoming_message(raw_msg: Dict[str, Any]) -> Dict[str, Any]:
    """Process a raw Telegram update and return structured message data.

    Returns:
        dict with keys: chat_id, user_id, message_id, text, image_data (optional)
        or empty dict if message should be ignored.
    """
    msg = raw_msg.get("message") or raw_msg.get("edited_message")
    if not msg:
        return {}

    chat_id = msg.get("chat", {}).get("id")
    user_id = msg.get("from", {}).get("id")
    message_id = msg.get("message_id")

    if not (chat_id and user_id):
        return {}

    # --- Authorization check ---
    st = load_state()
    owner_id = st.get("owner_id", 0)
    allowed = st.get("allowed_user_ids", [])

    if user_id != owner_id and user_id not in allowed:
        log.info("Ignored message from unauthorized user_id=%d", user_id)
        return {}

    # --- Extract text and image ---
    text_raw = (msg.get("text") or msg.get("caption") or "").strip()
    image_data = None

    if "photo" in msg:
        photos = msg["photo"]
        if photos:
            largest = max(photos, key=lambda p: p.get("file_size", 0))
            file_id = largest.get("file_id")
            if file_id:
                tg = get_tg()
                b64, mime = tg.download_file_base64(file_id, max_bytes=10_000_000)
                if b64:
                    image_data = {"base64": b64, "mime_type": mime}

    if not text_raw and not image_data:
        return {}

    # --- Log chat message ---
    log_chat("in", chat_id, user_id, text_raw)

    # --- Session tracking (для портретного анализа) ---
    session = get_user_session(user_id)
    if not session:
        start_user_session(user_id, message_id)
    # Always update session with current message (fixes first-message loss bug)
    update_user_session(user_id, message_id, text_raw)

    # Check portrait trigger (runs in background if >= 3 messages)
    check_portrait_trigger(user_id)

    return {
        "chat_id": chat_id,
        "user_id": user_id,
        "message_id": message_id,
        "text": text_raw,
        "image_data": image_data,
    }
