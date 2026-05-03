"""Detect question patterns in Claude's response and offer one-tap replies.

When Claude ends a turn with a yes/no question, we attach an inline keyboard
so the user can answer in one tap rather than typing 'yes'/'no'. This is a
pure heuristic — false positives are silent (just an extra button row that
doesn't fit the question), false negatives degrade to the normal text flow.
"""
from __future__ import annotations

import re
from typing import List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# Patterns we look at the *tail* of the message for. The tail is the last
# 400 chars; questions tend to come at the end.
_TAIL_BUDGET = 400

# Trigger when the tail ends with '?' AND contains one of these prompts.
_YES_NO_PATTERNS = [
    re.compile(r"\b(should I|shall I|do you want|do you'd like|would you like|may I|can I|ok to)\b", re.IGNORECASE),
    re.compile(r"\bproceed\b", re.IGNORECASE),
    re.compile(r"\(y/n\)", re.IGNORECASE),
    re.compile(r"\byes\s*/\s*no\b", re.IGNORECASE),
    re.compile(r"\bcontinue\?\s*$", re.IGNORECASE),
    re.compile(r"\bgo ahead\?\s*$", re.IGNORECASE),
]


def detect_quick_replies(response_text: str) -> Optional[InlineKeyboardMarkup]:
    """Return a keyboard if the response ends with a yes/no question, else None."""
    if not response_text:
        return None
    tail = response_text[-_TAIL_BUDGET:].strip()
    if not tail.endswith("?") and not tail.endswith("?**") and not tail.endswith('?"'):
        return None
    if not any(p.search(tail) for p in _YES_NO_PATTERNS):
        return None
    # Default keyboard for yes/no.
    rows: List[List[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton("✓ Yes", callback_data="quick:yes"),
            InlineKeyboardButton("✗ No", callback_data="quick:no"),
            InlineKeyboardButton("… Tell me more", callback_data="quick:more"),
        ]
    ]
    return InlineKeyboardMarkup(rows)
