# Architecture of Staff Portrait Analysis

## Triggers and Flow
### When Analysis is Triggered:
**Automatic Trigger** — function `check_portrait_trigger()` in `supervisor/portrait.py`:
```python
def check_portrait_trigger(user_id: int) -> None:
    session = get_user_session(user_id)
    if session and session["message_count"] >= 3:
        # Start portrait analysis in background thread
        background_thread.start(...)
```
### Conditions:
- ✅ Conversation >= 3 messages from user
- ✅ Triggers automatically in background thread (does not block dialogue)

## Session Reading and Message Handling
### Session Tracking (state.py):
**Two Points of Storage:**
1. **In-memory state** (`state.json`)
2. **Persistent storage** (e.g., Drive files)

### Saving Messages:
Messages from each session are saved with:
- User ID
- Message Text
- Timestamp

Messages are appended to the user’s session logs. Note: only `total_messages` will be counted for analysis thresholds.