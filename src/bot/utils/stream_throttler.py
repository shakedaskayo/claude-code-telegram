"""Coalesce frequent stream updates into Telegram-rate-friendly edit_text calls.

Telegram throttles editMessageText at roughly 1 request per second per chat. The
Claude SDK can emit far more stream events than that on tool-heavy work. Sending
one edit per event causes bursty, delayed updates because python-telegram-bot
queues the requests and they arrive late.

This throttler:
  * Stores only the latest pending text — older drafts are discarded.
  * Triggers an edit at most once per ``min_interval_s`` (default 1.0s).
  * Skips edits when the new text is identical to what was last sent.
  * Exposes ``flush()`` so the caller can force the final update through after
    the stream completes (the user must always see the final state).
  * Catches Telegram ``RetryAfter`` exceptions, doubles the interval temporarily,
    and never raises — a misbehaving Telegram path can never break the SDK loop.

Usage:

    throttler = StreamThrottler(progress_msg, min_interval_s=1.0)
    async def stream_handler(update_obj):
        text = await _format_progress_update(update_obj)
        if text:
            await throttler.update(text)
    ...
    await throttler.flush()
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


class StreamThrottler:
    """Rate-limit edit_text calls to a single Telegram message."""

    def __init__(
        self,
        message: Any,
        *,
        min_interval_s: float = 1.0,
        parse_mode: str = "HTML",
    ) -> None:
        self._message = message
        self._min_interval_s = min_interval_s
        self._parse_mode = parse_mode

        self._pending_text: Optional[str] = None
        self._last_sent_text: Optional[str] = None
        self._last_sent_at: float = 0.0
        # Backoff multiplier raised by Telegram RetryAfter; decays back to 1.0
        # on each successful send.
        self._backoff: float = 1.0

        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def update(self, text: str) -> None:
        """Record a new latest text and trigger a (possibly delayed) flush."""
        if not text:
            return
        async with self._lock:
            self._pending_text = text
            self._schedule_flush_locked()

    async def flush(self) -> None:
        """Force the latest pending text through immediately, bypassing throttle.

        Always called once at the end of a stream so the final state is visible.
        Safe to call when nothing is pending — becomes a no-op.
        """
        async with self._lock:
            if self._flush_task is not None and not self._flush_task.done():
                self._flush_task.cancel()
                self._flush_task = None
            await self._send_if_changed_locked()

    # --- internals ----------------------------------------------------------

    def _schedule_flush_locked(self) -> None:
        # Already a pending flush task — let it fire.
        if self._flush_task is not None and not self._flush_task.done():
            return
        delay = self._delay_until_next_send()
        if delay <= 0:
            # Eligible to send immediately; do it via a 0-delay task to keep
            # locking semantics simple (the lock is currently held).
            self._flush_task = asyncio.create_task(self._delayed_send(0))
        else:
            self._flush_task = asyncio.create_task(self._delayed_send(delay))

    def _delay_until_next_send(self) -> float:
        elapsed = time.monotonic() - self._last_sent_at
        target = self._min_interval_s * self._backoff
        return max(0.0, target - elapsed)

    async def _delayed_send(self, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            async with self._lock:
                await self._send_if_changed_locked()
        except asyncio.CancelledError:
            # flush() cancelled us — do nothing; flush() will send.
            pass

    async def _send_if_changed_locked(self) -> None:
        text = self._pending_text
        if text is None or text == self._last_sent_text:
            return
        try:
            await self._message.edit_text(text, parse_mode=self._parse_mode)
            self._last_sent_text = text
            self._last_sent_at = time.monotonic()
            self._backoff = 1.0
        except Exception as e:  # noqa: BLE001 - never let Telegram crash the stream
            self._handle_send_error(e)

    def _handle_send_error(self, e: Exception) -> None:
        """Log + adapt to common Telegram errors. Never re-raises."""
        msg = str(e)
        # python-telegram-bot raises RetryAfter for HTTP 429. The exception
        # object has a `.retry_after` attribute. We don't import it directly to
        # avoid coupling this util to a specific PTB version.
        retry_after = getattr(e, "retry_after", None)
        if retry_after:
            # Bump backoff so subsequent flushes throttle harder. Cap at 8x to
            # prevent cumulative drift if the chat is genuinely overloaded.
            self._backoff = min(8.0, self._backoff * 2)
            self._last_sent_at = time.monotonic() + float(retry_after) - self._min_interval_s
            logger.warning(
                "Telegram retry-after, backing off",
                retry_after=retry_after,
                new_backoff=self._backoff,
            )
            return
        # "message is not modified" is benign — our duplicate-text check should
        # already prevent it, but Telegram is sometimes pickier than us.
        if "not modified" in msg.lower():
            self._last_sent_text = self._pending_text
            return
        logger.warning("StreamThrottler edit_text failed", error=msg)
