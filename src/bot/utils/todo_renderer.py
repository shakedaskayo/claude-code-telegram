"""Render Claude's TodoWrite output as a structured Telegram message.

When the agent calls ``TodoWrite``, the bot looks up (or creates) a
'📋 Tasks' message for this turn and edits it in place to reflect the
latest task list. One Tasks message per turn — easy to scroll back to.

Status mapping:
    completed   → ✓
    in_progress → ⏳
    pending     → ○
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TodoTracker:
    """Owns the 📋 Tasks message for one turn — or for a session when
    backed by a SessionTracker (todo_message persists across iterations)."""

    chat: Any
    session: Any = None  # optional SessionTracker
    message: Any = None
    last_text: Optional[str] = None

    @property
    def _stored_message(self) -> Any:
        if self.session is not None:
            return self.session.todo_message
        return self.message

    @_stored_message.setter
    def _stored_message(self, value: Any) -> None:
        if self.session is not None:
            self.session.todo_message = value
        else:
            self.message = value

    @property
    def _stored_text(self) -> Optional[str]:
        if self.session is not None:
            return self.session.todo_last_text
        return self.last_text

    @_stored_text.setter
    def _stored_text(self, value: Optional[str]) -> None:
        if self.session is not None:
            self.session.todo_last_text = value
        else:
            self.last_text = value

    async def update(self, todos: List[dict]) -> None:
        """Render the todos and post or edit the message.

        When a SessionTracker is attached, the message reference lives on
        the session so the same '📋 Tasks' message is edited across all
        iterations (the user keeps scrolling back to the same message
        instead of accumulating a new one per turn)."""
        text = render_todos(todos)
        if not text:
            return
        msg = self._stored_message
        if msg is None:
            try:
                new_msg = await self.chat.send_message(text, parse_mode="HTML")
                self._stored_message = new_msg
                self._stored_text = text
            except Exception:  # noqa: BLE001
                self._stored_message = None
            return
        if text == self._stored_text:
            return
        try:
            await msg.edit_text(text, parse_mode="HTML")
            self._stored_text = text
        except Exception:  # noqa: BLE001
            # Telegram says 'message not modified' or rate-limits — ignore.
            pass


def render_todos(todos: List[dict]) -> str:
    """Pure function: list of todos → Telegram HTML message text.

    Each todo is expected to have:
        content: str
        status: 'pending' | 'in_progress' | 'completed'
        activeForm: str (optional, present-tense for in_progress display)
    """
    if not todos:
        return ""

    lines = ["📋 <b>Tasks</b>"]
    for t in todos:
        if not isinstance(t, dict):
            continue
        status = (t.get("status") or "").lower()
        icon = {
            "completed": "✓",
            "in_progress": "⏳",
            "pending": "○",
        }.get(status, "○")
        # Use activeForm for in_progress so it reads like a present-tense
        # action ("Updating gateways.rs"). Otherwise use content.
        body = t.get("content") or ""
        if status == "in_progress" and t.get("activeForm"):
            body = t["activeForm"]
        if not body:
            continue
        # Inline tag for in-progress so it stands out.
        if status == "in_progress":
            lines.append(f"{icon} <b>{_escape(body)}</b>")
        elif status == "completed":
            lines.append(f"{icon} <s>{_escape(body)}</s>")
        else:
            lines.append(f"{icon} {_escape(body)}")
    return "\n".join(lines)


def _escape(text: str) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
