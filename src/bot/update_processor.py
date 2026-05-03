"""Selective-concurrency update processor for PTB.

Regular updates process sequentially **per chat** — one at a time within a chat,
so the bot's per-session state stays coherent — but **different chats run in
parallel**. Priority callbacks (``stop:*``) bypass the queue entirely so they
can interrupt the currently-running handler.

Why per-chat locks instead of one global lock?
  Until this change, a single ``asyncio.Lock`` serialized every non-priority
  update. A 5-minute task in chat A blocked every message in chat B for those
  5 minutes. The intent of the lock was to make Stop callbacks work, not to
  enforce global single-tenancy — the per-chat version preserves the Stop
  behavior while letting independent conversations actually run independently.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any, Awaitable, Optional

import structlog
from telegram import Update
from telegram.ext._baseupdateprocessor import BaseUpdateProcessor

logger = structlog.get_logger()

# Cap how many per-chat locks we keep in memory. The bot is typically used by
# a single user but the limit guards against runaway growth in a long-lived
# process or in a misconfigured deployment.
_MAX_TRACKED_CHATS = 128

# Seconds since last use after which we may evict an idle lock during pruning.
_LOCK_IDLE_TTL_S = 3600


class StopAwareUpdateProcessor(BaseUpdateProcessor):
    """Per-chat sequential processing with priority bypass for stop callbacks.

    PTB calls ``process_update(update, coroutine)`` for every incoming update.
    The base class holds a semaphore (max 256) then calls our
    ``do_process_update()``.

    Routing:
      * Priority callbacks (``stop:*``): no lock, run immediately.
      * Everything else: acquire the lock for that chat's id. Different chats
        do not block each other; same chat is still strictly sequential.

    Stop-callback flow is unchanged: a stop callback arrives while a text
    handler holds the chat's lock -> stop runs concurrently because it bypasses
    -> fires the ``asyncio.Event`` -> the watcher task inside
    ``execute_command()`` calls ``client.interrupt()`` -> Claude stops ->
    ``run_command()`` returns -> handler finishes -> lock released.
    """

    _PRIORITY_PREFIXES = ("stop:",)

    def __init__(self) -> None:
        # High limit so priority callbacks are never blocked by semaphore
        super().__init__(max_concurrent_updates=256)

        # Per-chat locks. OrderedDict so we can prune the LRU tail.
        self._locks: "OrderedDict[int, asyncio.Lock]" = OrderedDict()
        # Last-use timestamp per chat for TTL eviction.
        self._last_used: dict[int, float] = {}
        # Protects the lock map itself when allocating / pruning.
        self._map_lock = asyncio.Lock()

    @classmethod
    def _is_priority_callback(cls, update: object) -> bool:
        """Return True if the update is a priority callback query."""
        if not isinstance(update, Update):
            return False
        cb = update.callback_query
        return (
            cb is not None
            and cb.data is not None
            and cb.data.startswith(cls._PRIORITY_PREFIXES)
        )

    @classmethod
    def _chat_id_for(cls, update: object) -> Optional[int]:
        """Best-effort chat id for routing. None falls back to a global lock."""
        if not isinstance(update, Update):
            return None
        # ``effective_chat`` covers messages, callbacks, edited messages,
        # channel posts, etc. — it's the right granularity here.
        chat = update.effective_chat
        return chat.id if chat is not None else None

    async def _get_lock(self, chat_id: Optional[int]) -> asyncio.Lock:
        """Return (and lazily create) the lock for this chat. Updates LRU order."""
        # Treat 'no chat id' as a single shared bucket so we don't crash, but
        # also don't accidentally widen the parallelism scope. In practice this
        # branch is rare (priority callbacks bypass already).
        key = chat_id if chat_id is not None else 0
        async with self._map_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
                self._maybe_prune_locked()
            else:
                # Touch LRU position
                self._locks.move_to_end(key)
            self._last_used[key] = time.monotonic()
            return lock

    def _maybe_prune_locked(self) -> None:
        """Evict idle locks once the map exceeds the cap.

        Caller must hold ``_map_lock``. We never evict a held lock — those are
        always 'in use' and the lookup will recreate them if missing anyway.
        """
        if len(self._locks) <= _MAX_TRACKED_CHATS:
            return
        now = time.monotonic()
        # Walk from the LRU side. Stop once we're back under cap.
        to_remove: list[int] = []
        for key, lock in self._locks.items():
            if len(self._locks) - len(to_remove) <= _MAX_TRACKED_CHATS:
                break
            if lock.locked():
                continue
            last = self._last_used.get(key, 0.0)
            if now - last < _LOCK_IDLE_TTL_S:
                # Honor the TTL — recently-used keys keep their lock.
                continue
            to_remove.append(key)
        for key in to_remove:
            self._locks.pop(key, None)
            self._last_used.pop(key, None)

    async def do_process_update(
        self,
        update: object,
        coroutine: Awaitable[Any],
    ) -> None:
        """Process an update, applying per-chat lock for non-priority updates."""
        if self._is_priority_callback(update):
            await coroutine
            return

        chat_id = self._chat_id_for(update)
        lock = await self._get_lock(chat_id)
        async with lock:
            await coroutine

    async def initialize(self) -> None:
        """Initialize the processor (no-op)."""

    async def shutdown(self) -> None:
        """Shutdown the processor (no-op)."""
