"""Stateful renderer for live Telegram progress messages.

The bot's stream_handler is called once per SDK event; each invocation needs to
produce a *current snapshot* string, not a delta, because Telegram's
``editMessageText`` always overwrites. ``ProgressRenderer`` accumulates state
across calls so the snapshot reads like a running monologue:

    🤖 Claude is working...

    > <last 240 chars of Claude's running text>

    Recent: ✓ Read foo.py · ✓ Bash:pytest · ⚙ Edit bar.py

The renderer handles:
  * ``stream_delta`` (content_block_delta): tokens streamed by Claude. We
    accumulate them so the user sees prose appear in near-real-time.
  * ``assistant`` with ``content``: a complete text block. We replace the
    accumulator with this exact text (it supersedes any partial deltas).
  * ``assistant`` with ``tool_calls``: tool calls about to execute. We render
    them in the recent-activity ribbon as ⚙ pending.
  * ``tool_result``: a tool finished. We mark its ribbon entry as ✓ or ❌.
  * ``thinking``: optional, not rendered (would be too noisy).
  * ``system`` init: rendered once at the very top.
  * ``error``: replaces the body with a clear error.

We use HTML mode and escape user content with ``html_format.escape_html``.
"""
from __future__ import annotations

import collections
from typing import Any, Optional

from .html_format import escape_html

# How many characters of Claude's running text to show. Telegram caps each
# message at 4096; we leave room for the ribbon and the prelude.
_TEXT_BUDGET = 1200
# Last N tool entries to keep in the ribbon. More than this and the message
# gets too tall; older entries roll off.
_RIBBON_KEEP = 6


class _ToolEntry:
    """One tool invocation tracked for the activity ribbon."""

    __slots__ = ("name", "summary", "status", "tool_use_id")

    def __init__(self, name: str, summary: str, tool_use_id: Optional[str]) -> None:
        self.name = name
        self.summary = summary
        self.status = "pending"  # pending | ok | fail
        self.tool_use_id = tool_use_id


class ProgressRenderer:
    """Stateful per-turn renderer."""

    def __init__(self) -> None:
        self._init_label: Optional[str] = None
        # Accumulating text from stream_delta events. Reset whenever we get a
        # complete AssistantMessage with content.
        self._running_text: str = ""
        # Most recent finalized text block (from assistant content). Replaces
        # running_text once received so we don't show stale deltas.
        self._final_text: Optional[str] = None
        # Activity ribbon: most-recent on the right, oldest on the left.
        self._tools: collections.deque[_ToolEntry] = collections.deque(
            maxlen=_RIBBON_KEEP
        )
        # Last error to surface, if any.
        self._error: Optional[str] = None

    # ------------------------------------------------------------------ feed

    def feed(self, update_obj: Any) -> None:
        """Update internal state from a StreamUpdate."""
        t = getattr(update_obj, "type", None)

        if t == "system":
            md = getattr(update_obj, "metadata", None) or {}
            if md.get("subtype") == "init":
                tools_count = len(md.get("tools", []))
                model = md.get("model", "Claude")
                self._init_label = f"🚀 {model} · {tools_count} tools"

        elif t == "stream_delta":
            content = getattr(update_obj, "content", None) or ""
            if content:
                self._running_text += content
                # Cap accumulator so we don't grow unbounded across a long turn.
                if len(self._running_text) > _TEXT_BUDGET * 4:
                    self._running_text = self._running_text[-_TEXT_BUDGET * 4 :]
                # New text supersedes the previous final block, since deltas
                # belong to a new block being built.
                self._final_text = None

        elif t == "assistant":
            content = getattr(update_obj, "content", None)
            tool_calls = getattr(update_obj, "tool_calls", None) or []
            if content:
                self._final_text = content
                self._running_text = ""  # superseded
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name") or "tool"
                summary = self._summarize_tool_input(name, tc.get("input"))
                self._tools.append(
                    _ToolEntry(name=name, summary=summary, tool_use_id=tc.get("id"))
                )

        elif t == "tool_result":
            md = getattr(update_obj, "metadata", None) or {}
            tool_use_id = md.get("tool_use_id")
            is_error = bool(getattr(update_obj, "is_error", lambda: False)())
            new_status = "fail" if is_error else "ok"
            # Mark the matching ribbon entry, or the most recent pending one
            # if we can't match by id.
            self._mark_tool_result(tool_use_id, new_status)

        elif t == "error":
            try:
                self._error = update_obj.get_error_message()
            except Exception:
                self._error = "An error occurred"

    # ---------------------------------------------------------------- render

    def render(self) -> Optional[str]:
        """Return the current snapshot. None means 'no change worth showing'."""
        # Decide what counts as the body text.
        body = self._final_text if self._final_text is not None else self._running_text
        body = body.strip()

        # Nothing useful to show yet?
        if not body and not self._tools and not self._error and not self._init_label:
            return None

        lines: list[str] = []

        if self._error:
            lines.append("❌ <b>Error</b>")
            lines.append("")
            lines.append(f"<i>{escape_html(self._error)}</i>")
            return "\n".join(lines)

        if self._init_label:
            lines.append(f"<i>{escape_html(self._init_label)}</i>")

        if body:
            lines.append("🤖 <b>Claude is working…</b>")
            lines.append("")
            # Show the tail of the running text — that's what's freshly
            # streaming in. Wrap with HTML quote-style indent for readability.
            shown = body[-_TEXT_BUDGET:]
            if len(body) > _TEXT_BUDGET:
                shown = "…" + shown
            lines.append(f"<blockquote>{escape_html(shown)}</blockquote>")
        elif self._tools:
            lines.append("🔧 <b>Using tools…</b>")

        if self._tools:
            ribbon_parts: list[str] = []
            for entry in self._tools:
                icon = self._status_icon(entry.status)
                label = entry.name
                if entry.summary:
                    label = f"{label}<code> {escape_html(entry.summary)}</code>"
                ribbon_parts.append(f"{icon} {label}")
            lines.append("")
            lines.append("<i>Recent:</i> " + " · ".join(ribbon_parts))

        return "\n".join(lines)

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _status_icon(status: str) -> str:
        return {"ok": "✓", "fail": "❌", "pending": "⚙"}.get(status, "•")

    def _mark_tool_result(self, tool_use_id: Optional[str], new_status: str) -> None:
        if tool_use_id:
            for entry in reversed(self._tools):
                if entry.tool_use_id == tool_use_id:
                    entry.status = new_status
                    return
        # Fallback: mark the most recent pending entry.
        for entry in reversed(self._tools):
            if entry.status == "pending":
                entry.status = new_status
                return

    @staticmethod
    def _summarize_tool_input(name: str, raw: Any) -> str:
        if not isinstance(raw, dict):
            return ""
        if name in {"Read", "Edit", "Write", "MultiEdit", "NotebookEdit", "NotebookRead"}:
            v = raw.get("file_path") or raw.get("path") or ""
            return _basename(v) if isinstance(v, str) else ""
        if name == "Bash":
            cmd = raw.get("command")
            return _truncate(cmd, 40) if isinstance(cmd, str) else ""
        if name == "Glob":
            v = raw.get("pattern")
            return _truncate(v, 40) if isinstance(v, str) else ""
        if name == "Grep":
            v = raw.get("pattern") or raw.get("query")
            return _truncate(v, 40) if isinstance(v, str) else ""
        if name in {"WebFetch", "WebSearch"}:
            v = raw.get("url") or raw.get("query")
            return _truncate(v, 40) if isinstance(v, str) else ""
        return ""


def _basename(path: str) -> str:
    s = path.rstrip("/")
    return s.rsplit("/", 1)[-1] or s


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"
