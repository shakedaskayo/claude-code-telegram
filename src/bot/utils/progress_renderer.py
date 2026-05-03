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
import time
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

    __slots__ = ("name", "summary", "status", "tool_use_id", "result_summary")

    def __init__(self, name: str, summary: str, tool_use_id: Optional[str]) -> None:
        self.name = name
        self.summary = summary
        self.status = "pending"  # pending | ok | fail
        self.tool_use_id = tool_use_id
        self.result_summary: Optional[str] = None


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
        # Heartbeat tracking: monotonic timestamps for "this turn started" and
        # "we last got any signal". The heartbeat task uses these to decide
        # when to force-render an updated elapsed counter even with no new
        # stream events.
        self._start_at: float = time.monotonic()
        self._last_event_at: float = self._start_at
        # Total tool invocations across the turn. _tools deque truncates so
        # this counter doesn't.
        self._total_tools: int = 0

    # ------------------------------------------------------------------ feed

    def feed(self, update_obj: Any) -> None:
        """Update internal state from a StreamUpdate."""
        # Always touch the heartbeat clock so the periodic refresher knows
        # whether the SDK is alive.
        self._last_event_at = time.monotonic()

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
                self._total_tools += 1

        elif t == "tool_result":
            md = getattr(update_obj, "metadata", None) or {}
            tool_use_id = md.get("tool_use_id")
            is_error = bool(md.get("is_error"))
            new_status = "fail" if is_error else "ok"
            result_text = getattr(update_obj, "content", None) or ""
            # Mark the matching ribbon entry and attach a short summary
            # extracted from the result text (line counts, exit codes, etc.).
            self._mark_tool_result(tool_use_id, new_status, result_text)

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

        # Nothing useful to show yet — return None so the throttler doesn't
        # post a placeholder. The pinned session tracker is already showing
        # 'iteration N · running' so the user knows the bot received them.
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

        # The pinned session tracker already owns the "what state am I in"
        # signal (running/idle/awaiting + elapsed + buttons). The per-turn
        # progress bubble's job is just to show streamed prose + tool ribbon
        # — no redundant 'Agent is working…' headline, no footer.
        if body:
            shown = body[-_TEXT_BUDGET:]
            if len(body) > _TEXT_BUDGET:
                shown = "…" + shown
            lines.append(f"<blockquote>{escape_html(shown)}</blockquote>")

        if self._tools:
            ribbon_parts: list[str] = []
            # Coalesce consecutive identical entries (same icon+name+summary)
            # into "label ×N". Result summaries (line counts, exit codes)
            # render after the input summary to give an at-a-glance sense of
            # what the tool actually did.
            run_label: Optional[str] = None
            run_count = 0
            for entry in self._tools:
                icon = self._status_icon(entry.status)
                label = entry.name
                if entry.summary:
                    label = f"{label}<code> {escape_html(entry.summary)}</code>"
                if entry.result_summary:
                    label = f"{label} <i>{escape_html(entry.result_summary)}</i>"
                full = f"{icon} {label}"
                if full == run_label:
                    run_count += 1
                else:
                    if run_label is not None:
                        ribbon_parts.append(
                            f"{run_label} ×{run_count}" if run_count > 1 else run_label
                        )
                    run_label = full
                    run_count = 1
            if run_label is not None:
                ribbon_parts.append(
                    f"{run_label} ×{run_count}" if run_count > 1 else run_label
                )
            lines.append("")
            lines.append("<i>Recent:</i> " + " · ".join(ribbon_parts))

        # Footer dropped — pinned session tracker shows elapsed/tools/state.
        # Keeping it here would duplicate that info in two places per iteration.

        return "\n".join(lines)

    # --- heartbeat helpers ------------------------------------------------

    def elapsed_seconds(self) -> int:
        return int(time.monotonic() - self._start_at)

    def idle_seconds(self) -> int:
        return int(time.monotonic() - self._last_event_at)

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _status_icon(status: str) -> str:
        return {"ok": "✓", "fail": "❌", "pending": "⚙"}.get(status, "•")

    def _mark_tool_result(
        self,
        tool_use_id: Optional[str],
        new_status: str,
        result_text: str = "",
    ) -> None:
        target: Optional[_ToolEntry] = None
        if tool_use_id:
            for entry in reversed(self._tools):
                if entry.tool_use_id == tool_use_id:
                    target = entry
                    break
        if target is None:
            # Fallback: mark the most recent pending entry.
            for entry in reversed(self._tools):
                if entry.status == "pending":
                    target = entry
                    break
        if target is None:
            return
        target.status = new_status
        target.result_summary = _extract_result_summary(target.name, result_text)

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


def _fmt_dur(seconds: int) -> str:
    """Format a positive duration for the heartbeat footer."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    h, rem = divmod(seconds, 3600)
    return f"{h}h {rem // 60}m"


def _extract_result_summary(tool_name: str, result_text: str) -> Optional[str]:
    """Extract a short stat from a tool's result content.

    Per-tool rules:
      Edit/Write/MultiEdit: parse "+N -N" if present, else None
      Read: count visible lines in the response
      Bash: extract exit code if present in output
      Glob/Grep: count matches (lines or 'N matches' phrasing)
      WebFetch: byte size
    Returns None when nothing useful can be derived.
    """
    if not result_text:
        return None
    text = result_text.strip()

    if tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        # Look for the SDK's standard '+N -N' style or count lines we can see.
        m = _PLUS_MINUS.search(text)
        if m:
            return m.group(0)
        # Sometimes the result is just the new content.
        line_count = text.count("\n") + 1
        if line_count > 0 and line_count < 10000:
            return f"{line_count} ln"
        return None

    if tool_name in ("Read", "NotebookRead"):
        # The CLI prefixes lines with "  N→" when reading files. Count those.
        line_match_count = sum(
            1 for ln in text.splitlines() if _LINE_PREFIX.match(ln)
        )
        if line_match_count:
            return f"{line_match_count} ln"
        # Fallback: raw line count (for non-file Reads).
        return f"{text.count(chr(10)) + 1} ln"

    if tool_name == "Bash":
        # CLI doesn't always include exit code; look for "exit code N".
        m = _EXIT_CODE.search(text)
        if m:
            return f"exit {m.group(1)}"
        # Heuristic: error-looking output → "errored"
        return None

    if tool_name in ("Glob", "Grep"):
        # The SDK formats result as "Found N matches" or just lists results.
        m = _MATCHES.search(text)
        if m:
            return f"{m.group(1)} match"
        # Otherwise count non-empty lines as a proxy.
        n = sum(1 for ln in text.splitlines() if ln.strip())
        if n:
            return f"{n} match"
        return None

    if tool_name in ("WebFetch", "WebSearch"):
        size = len(text)
        if size > 1024:
            return f"{size // 1024} KB"
        return f"{size}B"

    return None


# Regex helpers for result-summary extraction.
import re  # noqa: E402 (module-level constants near use site)
_PLUS_MINUS = re.compile(r"\+\d+\s+-\d+")
_LINE_PREFIX = re.compile(r"^\s*\d+→")
_EXIT_CODE = re.compile(r"exit\s+code[:\s]+(-?\d+)", re.IGNORECASE)
_MATCHES = re.compile(r"(\d+)\s+match", re.IGNORECASE)
