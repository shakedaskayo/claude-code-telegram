"""Interactive prompts: bridge between Claude SDK's can_use_tool callback
and Telegram's inline-keyboard UI.

When Claude calls AskUserQuestion (or, in a later commit, EnterPlanMode), we
need to:

1. Pause Claude's execution (the SDK awaits our can_use_tool result).
2. Post a Telegram message with inline keyboard buttons.
3. Wait for the user to tap a button (or send a free-text reply).
4. Return the answer to the SDK as a PermissionResultAllow with updated_input.

This module owns the wait/dispatch glue. The SDK side calls
``submit_question`` and awaits an asyncio.Future; the Telegram side calls
``resolve_question`` from a callback handler, which sets the future's result.
"""
from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

logger = structlog.get_logger()

# Each pending prompt expires after this long. The SDK call still continues
# polling, so a stale prompt would block Claude indefinitely; the timeout
# ensures we eventually un-stick.
PROMPT_TIMEOUT_S = 300

# Telegram callback_data has a 64-byte limit. We pack: <prefix>:<token>:<idx>
# where prefix is "ans" / "plan" / "voice" / "quick" and token is a short id.
_CALLBACK_PREFIXES = ("ans:", "plan:", "voice:", "quick:")


@dataclass
class _Selection:
    """Per-question selection state for multi-select questions."""

    question_index: int
    selected_indices: List[int] = field(default_factory=list)


@dataclass
class PendingPrompt:
    """One in-flight interactive prompt awaiting user response."""

    prompt_id: str
    user_id: int
    chat_id: int
    kind: str  # "ask" | "plan"
    # Original tool_input (we hand a possibly-modified copy back to Claude).
    tool_input: Dict[str, Any]
    # Future the SDK side awaits. Result is the dict to merge into
    # PermissionResultAllow's updated_input, or None to deny.
    future: "asyncio.Future[Optional[Dict[str, Any]]]"
    # The Telegram message we posted with the buttons (so we can edit it
    # to "Answered: X" once we have a result).
    prompt_message: Any = None
    # Multi-select state per question (only used when ask + multiSelect=True).
    selections: Dict[int, _Selection] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)


class InteractiveRegistry:
    """In-memory store of pending prompts, keyed by short id."""

    def __init__(self) -> None:
        self._prompts: Dict[str, PendingPrompt] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        user_id: int,
        chat_id: int,
        kind: str,
        tool_input: Dict[str, Any],
    ) -> Tuple[str, "asyncio.Future[Optional[Dict[str, Any]]]"]:
        prompt_id = secrets.token_urlsafe(6)
        loop = asyncio.get_event_loop()
        future: "asyncio.Future[Optional[Dict[str, Any]]]" = loop.create_future()
        prompt = PendingPrompt(
            prompt_id=prompt_id,
            user_id=user_id,
            chat_id=chat_id,
            kind=kind,
            tool_input=tool_input,
            future=future,
        )
        async with self._lock:
            self._prompts[prompt_id] = prompt
        return prompt_id, future

    async def get(self, prompt_id: str) -> Optional[PendingPrompt]:
        async with self._lock:
            return self._prompts.get(prompt_id)

    async def pop(self, prompt_id: str) -> Optional[PendingPrompt]:
        async with self._lock:
            return self._prompts.pop(prompt_id, None)

    async def find_for_user(self, user_id: int) -> Optional[PendingPrompt]:
        """Return the most recently registered prompt for this user, if any.
        Used so a free-text reply can be routed to a pending prompt without
        the user having to tap a 'custom' button first.
        """
        async with self._lock:
            candidates = [p for p in self._prompts.values() if p.user_id == user_id]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.created_at)

    async def expire_stale(self) -> None:
        now = time.monotonic()
        async with self._lock:
            stale = [
                pid for pid, p in self._prompts.items()
                if now - p.created_at > PROMPT_TIMEOUT_S
            ]
            for pid in stale:
                p = self._prompts.pop(pid, None)
                if p and not p.future.done():
                    logger.warning(
                        "Interactive prompt timed out",
                        prompt_id=pid,
                        user_id=p.user_id,
                        kind=p.kind,
                        age_s=now - p.created_at,
                    )
                    p.future.set_result(None)


# Singleton registry attached to bot_data on startup.
_REGISTRY_KEY = "interactive_registry"


def get_registry(context: ContextTypes.DEFAULT_TYPE) -> InteractiveRegistry:
    """Return the per-application registry, creating it on first access."""
    reg = context.bot_data.get(_REGISTRY_KEY)
    if reg is None:
        reg = InteractiveRegistry()
        context.bot_data[_REGISTRY_KEY] = reg
    return reg


def get_or_create_registry(bot_data: Dict[str, Any]) -> InteractiveRegistry:
    """Variant for callers without a Context (e.g., orchestrator setup)."""
    reg = bot_data.get(_REGISTRY_KEY)
    if reg is None:
        reg = InteractiveRegistry()
        bot_data[_REGISTRY_KEY] = reg
    return reg


# ---------------------------------------------------------------- AskUserQuestion

def build_ask_keyboard(
    prompt_id: str,
    questions: List[Dict[str, Any]],
    selections: Dict[int, _Selection],
) -> InlineKeyboardMarkup:
    """Render one keyboard for one question (the first unanswered one).

    For multi-select questions, options toggle on tap and a final 'Done'
    button submits. For single-select, taps submit immediately.

    Telegram limits inline keyboards: ~3 buttons per row, so we wrap.
    """
    # Find the first question that isn't fully answered yet. The data layer
    # tracks answers; if we're called for the first question only, we render
    # that one and the callback handler will pop it once submitted.
    qidx = _next_unanswered(questions, selections)
    if qidx is None:
        # All answered — defensive empty keyboard.
        return InlineKeyboardMarkup([])

    q = questions[qidx]
    opts = q.get("options") or []
    multi = bool(q.get("multiSelect"))
    sel = selections.get(qidx)
    selected = set(sel.selected_indices) if sel else set()

    rows: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    for i, opt in enumerate(opts):
        label = opt.get("label", f"Option {i + 1}")
        if multi:
            label = ("✓ " if i in selected else "○ ") + label
        # callback_data: ans:<prompt_id>:<qidx>:<oidx>
        data = f"ans:{prompt_id}:{qidx}:{i}"
        if len(data) > 60:
            # Shouldn't happen with our short ids but truncate just in case.
            data = data[:60]
        row.append(InlineKeyboardButton(label, callback_data=data))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    if multi:
        # Done button submits the current selections.
        rows.append([
            InlineKeyboardButton(
                "✓ done", callback_data=f"ans:{prompt_id}:{qidx}:done"
            )
        ])
    return InlineKeyboardMarkup(rows)


def _next_unanswered(
    questions: List[Dict[str, Any]],
    selections: Dict[int, _Selection],
) -> Optional[int]:
    """Index of the first question without a recorded selection. None if all answered."""
    for i, q in enumerate(questions):
        sel = selections.get(i)
        if sel is None:
            return i
        # multi-select question with empty selection is still "unanswered"
        # until the user taps Done — but we encode "done" by removing it from
        # selections and storing the answer in tool_input["answers"].
        # That logic lives in the callback handler; here we only need to know
        # whether to keep prompting.
    return None


def render_ask_text(
    questions: List[Dict[str, Any]],
    qidx: int,
    selections: Dict[int, _Selection],
) -> str:
    """Render the bot message text for the question being answered now."""
    q = questions[qidx]
    header = q.get("header") or "Question"
    text = q.get("question") or ""
    multi = bool(q.get("multiSelect"))

    parts = [f"🤔 <b>{_escape(header)}</b>", "", _escape(text)]

    # Show progress for multi-question prompts.
    if len(questions) > 1:
        parts.append("")
        parts.append(f"<i>question {qidx + 1} of {len(questions)}</i>")

    # Show currently-selected options for multi-select.
    if multi:
        sel = selections.get(qidx)
        chosen = sel.selected_indices if sel else []
        if chosen:
            opts = q.get("options") or []
            chosen_labels = [opts[i].get("label", "") for i in chosen if i < len(opts)]
            parts.append("")
            parts.append(f"<i>selected: {', '.join(_escape(c) for c in chosen_labels)}</i>")

    return "\n".join(parts)


def _escape(text: str) -> str:
    """Minimal HTML escape that matches the rest of the codebase."""
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ----------------------------------------------------------------- ExitPlanMode

def render_plan_text(plan: str) -> str:
    """Format the plan content for Telegram with a header."""
    body = _escape(plan).strip()
    if not body:
        body = "<i>(empty plan)</i>"
    return (
        "📋 <b>Proposed Plan</b>\n\n"
        f"<blockquote expandable>{body}</blockquote>"
    )


def build_plan_keyboard(prompt_id: str) -> InlineKeyboardMarkup:
    """Three-button keyboard for plan approval.

    callback_data formats:
      plan:<prompt_id>:approve
      plan:<prompt_id>:reject
      plan:<prompt_id>:modify
    """
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "✓ Approve", callback_data=f"plan:{prompt_id}:approve"
            ),
            InlineKeyboardButton(
                "✏ Modify", callback_data=f"plan:{prompt_id}:modify"
            ),
            InlineKeyboardButton(
                "✗ Reject", callback_data=f"plan:{prompt_id}:reject"
            ),
        ]
    ])
