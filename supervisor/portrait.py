"""
Supervisor — Portrait analysis module.

Generates psychological/professional portraits of staff (RecSys specialists)
based on conversation history using Claude Haiku.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

PORTRAIT_MODEL = "anthropic/claude-haiku-4-5"
OBSERVATIONS_MODEL = "anthropic/claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Knowledge base helpers
# ---------------------------------------------------------------------------

def _read_knowledge_file(drive_root: pathlib.Path, filename: str) -> str:
    """Read a knowledge file from Drive memory/knowledge/. Returns empty string on failure."""
    path = drive_root / "memory" / "knowledge" / filename
    if not path.exists():
        log.warning("Knowledge file not found: %s", path)
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        log.warning("Failed to read knowledge file: %s", path, exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# get_username
# ---------------------------------------------------------------------------

def get_username(user_id: int) -> str:
    """
    Get username for user_id from state.json.

    Returns username string or "user_{user_id}" as fallback.
    """
    from supervisor.state import load_state
    try:
        st = load_state()
        # Check user_sessions for stored username
        sessions = st.get("user_sessions", {})
        session = sessions.get(str(user_id), {})
        username = session.get("username")
        if username:
            return str(username)

        # Check allowed_user_ids metadata if available
        users_meta = st.get("users_meta", {})
        meta = users_meta.get(str(user_id), {})
        username = meta.get("username")
        if username:
            return str(username)
    except Exception:
        log.debug("Failed to load state for get_username(user_id=%s)", user_id, exc_info=True)

    return f"user_{user_id}"


# ---------------------------------------------------------------------------
# generate_observations
# ---------------------------------------------------------------------------

def generate_observations(
    user_id: int,
    messages: List[Dict[str, Any]],
    message_count: int,
) -> Dict[str, Any]:
    """
    Generate observations for a single conversation (not a full portrait).

    Args:
        user_id: Telegram user ID
        messages: List of {"role": "user/assistant", "content": str, "timestamp": str}
        message_count: Number of messages in this conversation

    Returns:
        Observations dict with structure from staff-portrait-prompt.md
    """
    from ouroboros.llm import LLMClient
    from supervisor.state import DRIVE_ROOT

    if not messages:
        log.warning("generate_observations called with empty messages for user_id=%s", user_id)
        return {"error": "no messages provided", "user_id": user_id}

    # Load knowledge base files
    portrait_prompt = _read_knowledge_file(DRIVE_ROOT, "staff-portrait-prompt.md")
    specialist_profile = _read_knowledge_file(DRIVE_ROOT, "recsys-specialist-profile.md")

    # Build system prompt focusing on observations only
    system_parts = []
    if portrait_prompt:
        system_parts.append(portrait_prompt)
        system_parts.append(
            "\n\nВАЖНО: Это анализ ОДНОГО разговора. "
            "Делай только наблюдения, НЕ строй портрет. "
            "Возвращай только структуру observations из раздела 'Как сохранять'."
        )
    else:
        system_parts.append(
            "Analyze this single conversation and return observations only (not a full portrait).\n"
            "Return JSON with structure: topics[], communication_notes, critical_thinking_moments[], "
            "accepted_without_verification[], thinking_style, patterns_noticed[], conversation_quality, confidence."
        )

    if specialist_profile:
        system_parts.append("\n\n---\n# RecSys Specialist Profile Reference\n\n" + specialist_profile)

    system_text = "\n".join(system_parts)

    # Format conversation
    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        ts = msg.get("timestamp", "")
        ts_prefix = f"[{ts}] " if ts else ""
        formatted_messages.append(f"{ts_prefix}{role.upper()}: {content}")

    conversation_text = "\n\n".join(formatted_messages)

    # Determine analysis level based on message count
    if message_count < 10:
        analysis_level = "базовые наблюдения (5-9 сообщений)"
    else:
        analysis_level = "расширенные наблюдения (10+ сообщений)"

    user_message = (
        f"Проанализируй разговор с user_id={user_id} ({message_count} сообщений).\n"
        f"Уровень анализа: {analysis_level}\n\n"
        f"Верни ТОЛЬКО валидный JSON структуры observations.\n"
        f"БЕЗ markdown fences, БЕЗ объяснений.\n\n"
        f"---\n# Разговор\n\n{conversation_text}"
    )

    # Build messages with prompt caching
    llm_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": user_message}],
        }
    ]

    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }

    all_messages = [system_message] + llm_messages

    try:
        client = LLMClient()
        response_msg, usage = client.chat(
            messages=all_messages,
            model=OBSERVATIONS_MODEL,
            max_tokens=2000,
            reasoning_effort="none",
        )
        raw_content = response_msg.get("content") or ""

        # Parse JSON
        observations_data = _parse_json_response(raw_content)
        observations_data["user_id"] = user_id
        observations_data["message_count"] = message_count
        observations_data["_model"] = OBSERVATIONS_MODEL
        observations_data["_usage"] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": usage.get("cached_tokens", 0),
            "cost": usage.get("cost", 0),
        }
        return observations_data

    except Exception:
        log.error("generate_observations failed for user_id=%s", user_id, exc_info=True)
        return {"error": "llm_call_failed", "user_id": user_id}


# ---------------------------------------------------------------------------
# generate_portrait
# ---------------------------------------------------------------------------

def generate_portrait(
    user_id: int,
    messages: List[Dict[str, Any]],
    previous_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a portrait of a staff member based on conversation messages.

    Args:
        user_id: Telegram user ID
        messages: List of {"role": "user/assistant", "content": str, "timestamp": str}
        previous_profile: Optional previous profile dict for context/dynamics

    Returns:
        Structured portrait dict as JSON from LLM analysis.
    """
    from ouroboros.llm import LLMClient
    from supervisor.state import DRIVE_ROOT

    if not messages:
        log.warning("generate_portrait called with empty messages for user_id=%s", user_id)
        return {"error": "no messages provided", "user_id": user_id}

    # Load knowledge base files
    portrait_prompt = _read_knowledge_file(DRIVE_ROOT, "staff-portrait-prompt.md")
    specialist_profile = _read_knowledge_file(DRIVE_ROOT, "recsys-specialist-profile.md")

    # Build system prompt
    system_parts = []
    if portrait_prompt:
        system_parts.append(portrait_prompt)
    else:
        system_parts.append(
            "You are an expert analyst specializing in professional and psychological profiling of RecSys specialists.\n"
            "Analyze the conversation and return a structured JSON portrait.\n"
            "The JSON must include fields: user_id, analysis_date, communication_style, "
            "technical_level, knowledge_areas, personality_traits, engagement_patterns, "
            "strengths, growth_areas, summary, confidence_score (0.0-1.0)."
        )
    if specialist_profile:
        system_parts.append("\n\n---\n# RecSys Specialist Profile Reference\n\n" + specialist_profile)

    system_text = "\n".join(system_parts)

    # Format conversation for analysis
    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        ts = msg.get("timestamp", "")
        ts_prefix = f"[{ts}] " if ts else ""
        formatted_messages.append(f"{ts_prefix}{role.upper()}: {content}")

    conversation_text = "\n\n".join(formatted_messages)

    # Build previous profile context if available
    prev_context = ""
    if previous_profile:
        try:
            prev_json = json.dumps(previous_profile, ensure_ascii=False, indent=2)
            prev_context = f"\n\n---\n# Previous Profile (use for tracking dynamics)\n\n{prev_json}"
        except Exception:
            log.debug("Failed to serialize previous_profile", exc_info=True)

    user_message = (
        f"Проанализируй разговор с user_id={user_id} и создай структурированный портрет.\n"
        f"Верни ТОЛЬКО валидный JSON без markdown fences и объяснений.\n"
        f"{prev_context}\n\n"
        f"ОБЯЗАТЕЛЬНАЯ СТРУКТУРА — строго соблюдать все поля:\n\n"
        f"{{\n"
        f"  \"user_id\": {user_id},\n"
        f"  \"portrait_status\": \"preliminary или full\",\n"
        f"  \"total_messages\": <число>,\n"
        f"  \"portrait\": {{\n"
        f"    \"request_patterns\": {{\n"
        f"      \"summary\": \"как обычно приходит к агенту\",\n"
        f"      \"recurring_themes\": [\"темы которые поднимает повторно\"],\n"
        f"      \"blind_zones\": [\"что никогда не спрашивает\"],\n"
        f"      \"examples\": [\"конкретные примеры из разговоров\"]\n"
        f"    }},\n"
        f"    \"thinking_quality\": {{\n"
        f"      \"summary\": \"как работает с информацией\",\n"
        f"      \"strengths\": [\"с конкретными примерами\"],\n"
        f"      \"weaknesses\": [\"с конкретными примерами — ОБЯЗАТЕЛЬНО минимум 2\"],\n"
        f"      \"examples\": [\"конкретные моменты\"]\n"
        f"    }},\n"
        f"    \"conversation_vector\": {{\n"
        f"      \"time_allocation\": \"на что тратит время\",\n"
        f"      \"movement\": \"куда движется судя по запросам\",\n"
        f"      \"avoided_topics\": [\"что избегает\"]\n"
        f"    }},\n"
        f"    \"dynamics\": {{\n"
        f"      \"summary\": \"как меняется со временем\",\n"
        f"      \"direction\": \"developing / static / regressing\",\n"
        f"      \"evidence\": [\"конкретные примеры динамики\"]\n"
        f"    }},\n"
        f"    \"key_insight\": \"острое наблюдение о паттерне мышления — не комплимент, с конкретным примером\"\n"
        f"  }},\n"
        f"  \"portrait_note\": \"статус и уровень уверенности\"\n"
        f"}}\n\n"
        f"ПРАВИЛА:\n"
        f"- thinking_quality.weaknesses ОБЯЗАТЕЛЬНЫ — минимум 2 пункта с примерами\n"
        f"- accepted_without_verification из разговора должны попасть в weaknesses\n"
        f"- key_insight — конкретный паттерн мышления, не похвала\n"
        f"- все поля обязательны, ничего не пропускать\n\n"
        f"---\n# Разговор\n\n{conversation_text}"
    )

    # Build messages with prompt caching on system message
    llm_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message,
                }
            ],
        }
    ]

    # System message with cache_control for prompt caching
    # Passed as first message with role "system" + cache_control on content block
    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }

    all_messages = [system_message] + llm_messages

    try:
        client = LLMClient()
        response_msg, usage = client.chat(
            messages=all_messages,
            model=PORTRAIT_MODEL,
            max_tokens=4000,
            reasoning_effort="none",
        )
        raw_content = response_msg.get("content") or ""

        # Parse JSON from response
        portrait_data = _parse_json_response(raw_content)
        portrait_data["user_id"] = user_id
        portrait_data["_model"] = PORTRAIT_MODEL
        portrait_data["_usage"] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": usage.get("cached_tokens", 0),
            "cost": usage.get("cost", 0),
        }
        return portrait_data

    except Exception:
        log.error("generate_portrait failed for user_id=%s", user_id, exc_info=True)
        return {"error": "llm_call_failed", "user_id": user_id}


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = raw.strip()

    # Strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```json or ```) and last line (```)
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        text = "\n".join(inner_lines).strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return {"data": result}
    except json.JSONDecodeError:
        log.warning("LLM returned non-JSON portrait response: %s", text[:200])
        return {"raw_response": text, "parse_error": "invalid_json"}


# ---------------------------------------------------------------------------
# save_conversation_log
# ---------------------------------------------------------------------------

def save_conversation_log(user_id: int, date: str, portrait_data: Dict[str, Any]) -> None:
    """
    Save conversation log to Drive: /logs/recsys/YYYY-MM-DD-[username].json

    Args:
        user_id: Telegram user ID
        date: Date string in YYYY-MM-DD format
        portrait_data: Portrait analysis result dict
    """
    from supervisor.state import DRIVE_ROOT

    username = get_username(user_id)
    filename = f"{date}-{username}.json"
    log_path = DRIVE_ROOT / "logs" / "recsys" / filename

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(portrait_data, ensure_ascii=False, indent=2)
        log_path.write_text(payload, encoding="utf-8")
        log.info("Saved conversation log: %s", log_path)
    except Exception:
        log.error("Failed to save conversation log for user_id=%s date=%s", user_id, date, exc_info=True)


# ---------------------------------------------------------------------------
# update_user_profile
# ---------------------------------------------------------------------------

def update_user_profile(user_id: int, new_portrait: Dict[str, Any]) -> None:
    """
    Read previous profile, merge new portrait entry with dynamics, save updated profile.

    Profile path: /profiles/[username].json
    Accumulates all portrait entries with dates.
    """
    from supervisor.state import DRIVE_ROOT
    import datetime

    username = get_username(user_id)
    profile_path = DRIVE_ROOT / "profiles" / f"{username}.json"

    # Load existing profile if it exists
    existing_profile: Dict[str, Any] = {}
    if profile_path.exists():
        try:
            existing_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if not isinstance(existing_profile, dict):
                existing_profile = {}
        except Exception:
            log.warning("Failed to read existing profile for user_id=%s, starting fresh", user_id, exc_info=True)
            existing_profile = {}

    # Initialize profile structure if new
    if not existing_profile:
        existing_profile = {
            "user_id": user_id,
            "username": username,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "entries": [],
            "dynamics": {},
        }

    # Append new portrait entry with timestamp
    entry = {
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "portrait": new_portrait,
    }
    entries: List[Dict[str, Any]] = existing_profile.setdefault("entries", [])
    entries.append(entry)

    # Update dynamics: track changes in key scalar fields across entries
    dynamics = existing_profile.setdefault("dynamics", {})
    tracked_fields = ("technical_level", "confidence_score", "engagement_patterns", "communication_style")
    for field in tracked_fields:
        value = new_portrait.get(field)
        if value is not None:
            history = dynamics.setdefault(field, [])
            history.append({"date": entry["date"], "value": value})

    existing_profile["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    existing_profile["username"] = username  # keep in sync

    try:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(existing_profile, ensure_ascii=False, indent=2)
        profile_path.write_text(payload, encoding="utf-8")
        log.info("Updated profile for user_id=%s at %s", user_id, profile_path)
    except Exception:
        log.error("Failed to save profile for user_id=%s", user_id, exc_info=True)


# ---------------------------------------------------------------------------
# check_portrait_trigger
# ---------------------------------------------------------------------------

def check_portrait_trigger(user_id: int) -> None:
    """
    Запускает анализ по порогам количества сообщений в отдельном потоке.

    Пороги:
      - <5 сообщений: ничего не делаем
      - >=5: базовые наблюдения (generate_observations, один раз за сессию)
      - >=10: полный анализ разговора (generate_observations, один раз за сессию)
      - >=15: preliminary-портрет (generate_portrait, один раз за сессию)
      - >=40: full-портрет, затем обновление каждые +20 сообщений
    """
    import threading

    def _run() -> None:
        try:
            log.info("Portrait analysis thread started for user_id=%s", user_id)
            from supervisor.state import (
                DRIVE_ROOT,
                get_user_session,
                acquire_file_lock,
                release_file_lock,
                STATE_LOCK_PATH,
                _load_state_unlocked,
                _save_state_unlocked,
            )
            import datetime

            # Read inbound messages for this user from chat.jsonl
            chat_log_path = DRIVE_ROOT / "logs" / f"chat_{user_id}.jsonl"
            messages: List[Dict[str, Any]] = []
            if chat_log_path.exists():
                with open(chat_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get("user_id") == user_id and entry.get("direction") == "in":
                                content = entry.get("text", "")
                                if content:
                                    messages.append({
                                        "role": "user",
                                        "content": content,
                                        "timestamp": entry.get("ts", ""),
                                    })
                        except Exception:
                            pass

            if not messages:
                log.debug("check_portrait_trigger: no messages found for user_id=%s", user_id)
                return

            # Текущее состояние сессии (счётчик сообщений и стадия анализа)
            session = get_user_session(user_id)
            if session is not None:
                try:
                    message_count = len(messages)
                except Exception:
                    message_count = len(messages)
                try:
                    portrait_stage = int(session.get("portrait_stage") or 0)
                except Exception:
                    portrait_stage = 0
                last_full_at = session.get("last_full_portrait_message_count")
                if isinstance(last_full_at, str):
                    try:
                        last_full_at = int(last_full_at)
                    except Exception:
                        last_full_at = None
                if not isinstance(last_full_at, int):
                    last_full_at = None
            else:
                message_count = len(messages)
                portrait_stage = 0
                last_full_at = None

            if message_count < 5:
                log.debug("check_portrait_trigger: message_count=%s < 5, skipping", message_count)
                return

            # Дата разговора по последнему сообщению (общая для всех сохранений)
            if messages:
                last_ts = messages[-1].get("timestamp", "")
                try:
                    dt = datetime.datetime.fromisoformat(last_ts.replace("+00:00", "+00:00"))
                    conversation_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    conversation_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
            else:
                conversation_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

            new_stage = portrait_stage
            new_last_full_at = last_full_at

            # --- >=5: базовые наблюдения (один раз за сессию) ---
            if message_count >= 5 and portrait_stage < 1:
                obs = generate_observations(user_id, messages, message_count)
                if "error" in obs:
                    log.warning("Basic observations failed for user_id=%s: %s",
                                user_id, obs.get("error"))
                else:
                    save_conversation_log(user_id, conversation_date, obs)
                    new_stage = max(new_stage, 1)

            # --- >=10: полный анализ (один раз за сессию) ---
            if message_count >= 10 and portrait_stage < 2:
                obs_full = generate_observations(user_id, messages, message_count)
                if "error" in obs_full:
                    log.warning("Full observations failed for user_id=%s: %s",
                                user_id, obs_full.get("error"))
                else:
                    save_conversation_log(user_id, conversation_date, obs_full)
                    new_stage = max(new_stage, 2)

            # Подготовим предыдущий профиль для портретов
            username = get_username(user_id)
            profile_path = DRIVE_ROOT / "profiles" / f"{username}.json"
            previous_profile: Optional[Dict[str, Any]] = None
            if profile_path.exists():
                try:
                    previous_profile = json.loads(profile_path.read_text(encoding="utf-8"))
                except Exception:
                    log.debug("Failed to load previous profile for user_id=%s", user_id, exc_info=True)

            # --- >=15 и <40: preliminary-портрет (один раз за сессию) ---
            if 15 <= message_count < 40 and portrait_stage < 3:
                portrait_data = generate_portrait(user_id, messages, previous_profile)
                if "error" in portrait_data:
                    log.warning("Preliminary portrait generation failed for user_id=%s: %s",
                                user_id, portrait_data.get("error"))
                else:
                    portrait_data.setdefault("portrait_kind", "preliminary")
                    save_conversation_log(user_id, conversation_date, portrait_data)
                    update_user_profile(user_id, portrait_data)
                    new_stage = max(new_stage, 3)

            # --- >=40: full-портрет + обновление каждые +20 сообщений ---
            should_run_full = False
            if message_count >= 40:
                if last_full_at is None:
                    should_run_full = True
                else:
                    if (message_count - last_full_at) >= 20:
                        should_run_full = True

            if should_run_full:
                full_portrait = generate_portrait(user_id, messages, previous_profile)
                if "error" in full_portrait:
                    log.warning("Full portrait generation failed for user_id=%s: %s",
                                user_id, full_portrait.get("error"))
                else:
                    full_portrait.setdefault("portrait_kind", "full")
                    save_conversation_log(user_id, conversation_date, full_portrait)
                    update_user_profile(user_id, full_portrait)
                    new_stage = max(new_stage, 4)
                    new_last_full_at = message_count

            # Обновляем метаданные сессии (но не обнуляем message_count)
            lock_fd = acquire_file_lock(STATE_LOCK_PATH)
            try:
                st = _load_state_unlocked()
                sessions = st.setdefault("user_sessions", {})
                sess = sessions.get(str(user_id))
                if sess is not None:
                    sess["message_count"] = message_count
                    if new_stage != portrait_stage:
                        sess["portrait_stage"] = new_stage
                    if new_last_full_at is not None:
                        sess["last_full_portrait_message_count"] = new_last_full_at
                    _save_state_unlocked(st)
            finally:
                release_file_lock(STATE_LOCK_PATH, lock_fd)

            log.info("Portrait analysis completed for user_id=%s (messages=%s, stage=%s)",
                     user_id, message_count, new_stage)

        except Exception:
            log.error("check_portrait_trigger background thread failed for user_id=%s",
                      user_id, exc_info=True)

    thread = threading.Thread(target=_run, daemon=True, name=f"portrait-{user_id}")
    thread.start()
    log.debug("check_portrait_trigger: background thread started for user_id=%s", user_id)
