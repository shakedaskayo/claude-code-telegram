"""Pinned 'Active Session' tracker.

ONE pinned message per Claude session (from /new to /new), surviving any
number of iterations (turns). Iterations are added to the same pinned
message — it doesn't reset on each follow-up.

State machine:
    running   → an iteration is producing events (≤30s since last)
    stalled   → no events for >30s; heartbeat still firing
    awaiting  → AskUserQuestion or ExitPlanMode pending
    idle      → between iterations; user hasn't sent next prompt yet
    completed → session ended cleanly (user sent /end or /new)
    failed    → fatal error in last iteration

Pinning notifies once at start; edits don't notify; we unpin only when
the session completes/ends.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

logger = structlog.get_logger()

State = str  # "running" | "stalled" | "awaiting" | "idle" | "completed" | "failed"

# How many recent iterations to keep visible in the expanded view.
_RECENT_ITERATIONS = 6


@dataclass
class _Iteration:
    """One Claude turn within a session."""

    index: int
    prompt_preview: str
    started_at: float
    ended_at: Optional[float] = None
    state: str = "running"  # "running" | "completed" | "failed"
    tools: int = 0
    cost: float = 0.0


@dataclass
class SessionTracker:
    """One per Claude session. Lives across iterations."""

    chat: Any
    user_id: int
    workspace: Optional[str] = None
    name: Optional[str] = None  # derived from first prompt
    started_at: float = field(default_factory=time.monotonic)
    state: State = "idle"
    detail: str = ""
    total_tools: int = 0
    total_cost: float = 0.0
    iterations: Deque[_Iteration] = field(default_factory=lambda: deque(maxlen=_RECENT_ITERATIONS))
    expanded: bool = False  # whether to render the verbose iteration list
    pinned: bool = False
    message: Any = None  # the pinned telegram Message
    last_event_at: float = field(default_factory=time.monotonic)
    interrupt_event: Optional[asyncio.Event] = None
    # Per-session todo message (persists across iterations).
    todo_message: Any = None
    todo_last_text: Optional[str] = None
    # Recent ribbon — last N tool calls / events for the at-a-glance pin.
    recent: Deque[str] = field(default_factory=lambda: deque(maxlen=5))

    @property
    def iteration_count(self) -> int:
        """All-time count, even past the deque maxlen."""
        return getattr(self, "_iteration_count", 0)

    @iteration_count.setter
    def iteration_count(self, value: int) -> None:
        self._iteration_count = value

    @property
    def current_iteration(self) -> Optional[_Iteration]:
        return self.iterations[-1] if self.iterations else None

    # --- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Post + pin the initial session message."""
        self._iteration_count = 0
        text = self._render()
        try:
            self.message = await self.chat.send_message(
                text, parse_mode="HTML", reply_markup=self._keyboard()
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to post session tracker", error=str(e))
            return
        try:
            await self.chat.pin_message(
                self.message.message_id, disable_notification=False
            )
            self.pinned = True
        except Exception as e:  # noqa: BLE001
            logger.debug("Pin failed, continuing unpinned", error=str(e))

    async def begin_iteration(self, prompt: str) -> None:
        """A new turn (follow-up message) is starting."""
        self._iteration_count = self.iteration_count + 1
        # Derive a session name from the first message we ever see.
        if self.name is None and prompt:
            self.name = _shorten(prompt.strip(), 50)
        it = _Iteration(
            index=self._iteration_count,
            prompt_preview=_shorten(prompt, 80),
            started_at=time.monotonic(),
        )
        self.iterations.append(it)
        self.state = "running"
        self.detail = ""
        self.last_event_at = time.monotonic()
        await self._refresh()

    async def transition(
        self,
        new_state: State,
        *,
        detail: Optional[str] = None,
        send_notification: bool = False,
    ) -> None:
        """Update state without ending the iteration."""
        self.state = new_state
        if detail is not None:
            self.detail = detail
        self.last_event_at = time.monotonic()
        await self._refresh()
        if send_notification and new_state == "awaiting":
            try:
                await self.chat.send_message(
                    "⏸ <b>Agent needs your input</b> — see the question above.",
                    parse_mode="HTML",
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Awaiting notification failed", error=str(e))

    async def end_iteration(
        self, *, success: bool, tools: int = 0, cost: float = 0.0
    ) -> None:
        """Mark the current iteration done. Session stays alive."""
        it = self.current_iteration
        if it is not None:
            it.ended_at = time.monotonic()
            it.state = "completed" if success else "failed"
            it.tools = tools
            it.cost = cost
        self.total_tools += tools
        self.total_cost += cost
        self.state = "idle"  # waiting for next user message
        self.detail = ""
        await self._refresh()

    async def end_session(self, *, success: bool = True) -> None:
        """Retire the pin. Called on /new or /end."""
        self.state = "completed" if success else "failed"
        await self._refresh(no_keyboard=True)
        if self.pinned and self.message is not None:
            try:
                await self.chat.unpin_message(self.message.message_id)
                self.pinned = False
            except Exception as e:  # noqa: BLE001
                logger.debug("Unpin failed", error=str(e))

    async def finish(self, success: bool, summary: Optional[str] = None) -> None:
        """Backwards-compatible alias: end the current iteration cleanly.

        The orchestrator's existing finally-block calls finish(success=...)
        once per Claude turn. With the per-session model, that's the end of
        an iteration, not the whole session. The session itself stays alive
        until the user sends /new or taps [End session].
        """
        if summary is not None and not self.detail:
            self.detail = summary
        await self.end_iteration(success=success)

    async def heartbeat_tick(self) -> None:
        """Detect running → stalled and refresh elapsed."""
        if self.state in ("completed", "failed", "idle"):
            return
        idle = time.monotonic() - self.last_event_at
        if self.state == "running" and idle > 30:
            self.state = "stalled"
        await self._refresh()

    def bump_tools(self, n: int = 1, label: Optional[str] = None) -> None:
        it = self.current_iteration
        if it is not None:
            it.tools += n
        self.total_tools += n
        self.last_event_at = time.monotonic()
        if label:
            self.recent.append(label)

    def bump_event(self) -> None:
        """Mark that an event happened (used to keep stalled away)."""
        self.last_event_at = time.monotonic()

    async def toggle_expanded(self) -> None:
        self.expanded = not self.expanded
        await self._refresh()

    # --- rendering --------------------------------------------------------

    async def _refresh(self, no_keyboard: bool = False) -> None:
        if self.message is None:
            return
        try:
            await self.message.edit_text(
                self._render(),
                parse_mode="HTML",
                reply_markup=None if no_keyboard else self._keyboard(),
            )
        except Exception:  # noqa: BLE001
            pass  # 'not modified' / rate-limit noise

    def _render(self) -> str:
        emoji = _STATE_EMOJI.get(self.state, "•")
        elapsed = int(time.monotonic() - self.started_at)
        ws = self.workspace or "—"

        # Header: session name (if known) + workspace + elapsed.
        if self.name:
            header = (
                f"📌 <b>{_escape(self.name)}</b>\n"
                f"<i>{_escape(ws)}</i> · {_fmt_dur(elapsed)}"
            )
        else:
            header = f"📌 <b>{_escape(ws)}</b> · session {_fmt_dur(elapsed)}"
        line2_parts = [f"{emoji} <i>{self.state}</i>"]
        if self._iteration_count:
            line2_parts.append(f"iteration {self._iteration_count}")
        if self.total_tools:
            line2_parts.append(f"🔧 {self.total_tools} tools")
        if self.total_cost > 0:
            line2_parts.append(f"💸 ${self.total_cost:.2f}")
        line2 = " · ".join(line2_parts)

        lines = [header, line2]

        # If currently working on something, show it + recent activity.
        if self.detail and self.state in ("running", "stalled", "awaiting"):
            lines.append("")
            lines.append(f"<i>Now:</i> {_escape(self.detail[:140])}")
        if self.recent and self.state in ("running", "stalled", "awaiting"):
            ribbon = " · ".join(_escape(s) for s in list(self.recent))
            lines.append(f"<i>Recent:</i> {ribbon}")

        # Expanded iteration list — opt-in via the [Show iterations] toggle.
        if self.expanded and self.iterations:
            lines.append("")
            lines.append("<i>Recent iterations:</i>")
            for it in self.iterations:
                its = _ITER_EMOJI.get(it.state, "•")
                dur = _fmt_dur(int((it.ended_at or time.monotonic()) - it.started_at))
                lines.append(
                    f"  {its} #{it.index} · {dur} · {it.tools}t — "
                    f"<i>{_escape(it.prompt_preview)}</i>"
                )

        return "\n".join(lines)

    def _keyboard(self) -> Optional[InlineKeyboardMarkup]:
        if self.state in ("completed", "failed"):
            return None
        # Compose the action row. We keep it tight (3 buttons) plus an
        # expand toggle on a second row when there's something to expand.
        rows = []
        active = self.state in ("running", "stalled", "awaiting")
        action_row = []
        if active:
            action_row.append(
                InlineKeyboardButton("■ Stop", callback_data=f"stop:{self.user_id}")
            )
        action_row.append(
            InlineKeyboardButton("📥 Queue", callback_data=f"trk:queue:{self.user_id}")
        )
        action_row.append(
            InlineKeyboardButton("ℹ Status", callback_data=f"trk:status:{self.user_id}")
        )
        rows.append(action_row)
        toggle_label = "▴ Hide iterations" if self.expanded else "▾ Show iterations"
        rows.append([
            InlineKeyboardButton(toggle_label, callback_data=f"trk:expand:{self.user_id}"),
            InlineKeyboardButton(
                "✕ End session", callback_data=f"trk:end:{self.user_id}"
            ),
        ])
        return InlineKeyboardMarkup(rows)


_STATE_EMOJI: Dict[str, str] = {
    "running": "🟢",
    "stalled": "🟡",
    "awaiting": "🔵",
    "idle": "⚪",
    "completed": "✓",
    "failed": "❌",
}
_ITER_EMOJI: Dict[str, str] = {
    "running": "🟢",
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


def _shorten(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


# ============================================================ Queued prompts

_QUEUED: Dict[int, Dict[str, Any]] = {}
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


# ============================================================ TaskTracker compat

# Some older code paths import TaskTracker from this module. Keep an alias so
# callers don't break while the rename is in flight.
TaskTracker = SessionTracker
