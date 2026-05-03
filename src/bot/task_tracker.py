"""Pinned 'Active Task' status message lifecycle.

The bot maintains one pinned message per turn so the user always knows what's
happening — even when they scroll up, even when typing indicators die, even
when the progress bubble is far below.

State machine:
    running   → Claude is actively producing events (≤30s since last)
    stalled   → no events for >30s; heartbeat still firing
    awaiting  → AskUserQuestion or ExitPlanMode pending
    paused    → user tapped Pause (not currently exposed via button; future)
    completed → turn finished cleanly; auto-unpin in 30s
    failed    → turn errored; stays pinned until acknowledged

Only state transitions edit the pinned message. The body shows a small,
glanceable summary; tap [Status] to get a verbose detail message.

Pinning notifies once at start; edits don't notify. We unpin on completion.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

logger = structlog.get_logger()

State = str  # "running" | "stalled" | "awaiting" | "paused" | "completed" | "failed"

# Auto-unpin completed tasks after this delay so the chat isn't cluttered
# with old "✓ Done" pins. Failed tasks stay pinned until the user starts a
# new one (we'll unpin then).
_AUTO_UNPIN_DELAY_S = 30


@dataclass
class TaskTracker:
    """One per turn. Owns the pinned status message + lifecycle."""

    chat: Any
    user_id: int
    workspace: Optional[str] = None
    started_at: float = field(default_factory=time.monotonic)
    state: State = "running"
    last_event_at: float = field(default_factory=time.monotonic)
    detail: str = ""
    tool_count: int = 0
    pinned: bool = False
    message: Any = None  # the pinned (or fallback) telegram Message

    # Set by the orchestrator after construction so action buttons can talk
    # back to the running turn.
    interrupt_event: Optional[asyncio.Event] = None

    async def start(self) -> None:
        """Post the initial status message and try to pin it."""
        text = self._render()
        try:
            self.message = await self.chat.send_message(
                text,
                parse_mode="HTML",
                reply_markup=self._keyboard(),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to post task tracker", error=str(e))
            return
        # Pin best-effort. In private chats we usually have can_pin_messages.
        try:
            await self.chat.pin_message(
                self.message.message_id, disable_notification=False
            )
            self.pinned = True
        except Exception as e:  # noqa: BLE001
            # Permissions or transient — fall back to the unpinned 'sticky' feel.
            logger.debug("Pin failed, continuing unpinned", error=str(e))

    async def transition(
        self,
        new_state: State,
        *,
        detail: Optional[str] = None,
        tool_count: Optional[int] = None,
        send_notification: bool = False,
    ) -> None:
        """Move to a new state and refresh the pinned message."""
        if new_state == self.state and detail is None and tool_count is None:
            return  # no-op
        self.state = new_state
        if detail is not None:
            self.detail = detail
        if tool_count is not None:
            self.tool_count = tool_count
        self.last_event_at = time.monotonic()

        if self.message is not None:
            try:
                await self.message.edit_text(
                    self._render(),
                    parse_mode="HTML",
                    reply_markup=self._keyboard(),
                )
            except Exception as e:  # noqa: BLE001 - never let UI crash the turn
                logger.debug("Tracker edit skipped", error=str(e))

        # Awaiting-input is the one transition we always notify about, since
        # it means the agent literally cannot continue without the user.
        if send_notification and new_state == "awaiting":
            try:
                await self.chat.send_message(
                    "⏸ <b>Agent needs your input</b> — see the question above.",
                    parse_mode="HTML",
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Awaiting notification failed", error=str(e))

    async def heartbeat_tick(self) -> None:
        """Called periodically by the orchestrator's progress heartbeat.

        Detects 'stalled' (no events for 30s while running) and edits the
        pinned message so the user sees the elapsed counter advance.
        """
        if self.state in ("completed", "failed"):
            return
        idle = time.monotonic() - self.last_event_at
        if self.state == "running" and idle > 30:
            await self.transition("stalled")
            return
        # Just refresh the elapsed counter.
        if self.message is not None:
            try:
                await self.message.edit_text(
                    self._render(),
                    parse_mode="HTML",
                    reply_markup=self._keyboard(),
                )
            except Exception:  # noqa: BLE001
                pass  # noise: 'message is not modified' / rate limits

    async def finish(self, success: bool, summary: Optional[str] = None) -> None:
        """Mark the task done and arrange for auto-unpin."""
        self.state = "completed" if success else "failed"
        if summary:
            self.detail = summary
        if self.message is not None:
            try:
                await self.message.edit_text(
                    self._render(),
                    parse_mode="HTML",
                    reply_markup=None,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Finish edit skipped", error=str(e))
        # Successful tasks auto-unpin after a delay; failed ones stay pinned.
        if success and self.pinned:
            asyncio.create_task(self._auto_unpin())

    async def _auto_unpin(self) -> None:
        try:
            await asyncio.sleep(_AUTO_UNPIN_DELAY_S)
            if self.message is not None:
                await self.chat.unpin_message(self.message.message_id)
                self.pinned = False
        except Exception as e:  # noqa: BLE001
            logger.debug("Auto-unpin failed", error=str(e))

    # --- rendering --------------------------------------------------------

    def _render(self) -> str:
        elapsed = int(time.monotonic() - self.started_at)
        emoji = _STATE_EMOJI.get(self.state, "•")
        state_label = self.state
        ws = self.workspace or "—"

        lines = [
            f"📌 <b>Active Task</b> — {_escape(ws)}",
            f"{emoji} <i>{state_label}</i> · ⏱ {_fmt_dur(elapsed)}"
            + (f" · 🔧 {self.tool_count} tools" if self.tool_count else ""),
        ]
        if self.detail:
            lines.append(f"<i>{_escape(self.detail[:140])}</i>")
        return "\n".join(lines)

    def _keyboard(self) -> Optional[InlineKeyboardMarkup]:
        if self.state in ("completed", "failed"):
            return None
        # callback_data fits within Telegram's 64-byte limit easily.
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "■ Stop", callback_data=f"stop:{self.user_id}"
                ),
                InlineKeyboardButton(
                    "📥 Queue", callback_data=f"trk:queue:{self.user_id}"
                ),
                InlineKeyboardButton(
                    "ℹ Status", callback_data=f"trk:status:{self.user_id}"
                ),
            ]
        ])


_STATE_EMOJI: Dict[str, str] = {
    "running": "🟢",
    "stalled": "🟡",
    "awaiting": "🔵",
    "paused": "⏸",
    "completed": "✓",
    "failed": "❌",
}


def _fmt_dur(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    h, rem = divmod(seconds, 3600)
    return f"{h}h {rem // 60}m"


def _escape(text: str) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ============================================================ Queued prompts

# In-memory store: user_id -> {"text": str, "delivered_at": float}.
# When a turn finishes, the orchestrator drains this for the user and posts
# the queued text as if they'd just sent it.
_QUEUED: Dict[int, Dict[str, Any]] = {}
# user_id -> True while awaiting their next text after tapping [Queue].
_AWAITING_QUEUE_TEXT: Dict[int, bool] = {}


def queue_pending(user_id: int) -> bool:
    return _AWAITING_QUEUE_TEXT.get(user_id, False)


def mark_queue_pending(user_id: int) -> None:
    _AWAITING_QUEUE_TEXT[user_id] = True


def clear_queue_pending(user_id: int) -> None:
    _AWAITING_QUEUE_TEXT.pop(user_id, None)


def store_queued_prompt(user_id: int, text: str) -> None:
    _QUEUED[user_id] = {"text": text, "ts": time.time()}
    clear_queue_pending(user_id)


def take_queued_prompt(user_id: int) -> Optional[str]:
    item = _QUEUED.pop(user_id, None)
    if item is None:
        return None
    return item.get("text")
