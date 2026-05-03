"""Telegram callback handler for interactive prompts.

Handles ``ans:*``, ``plan:*``, ``voice:*``, and ``quick:*`` callback_data
prefixes. For commit 1 we wire ``ans:*`` (AskUserQuestion); the others land
in subsequent commits but the dispatch is structured so adding them is a
small extension.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from telegram import Update
from telegram.ext import ContextTypes

from ..interactive import (
    PendingPrompt,
    _Selection,
    build_ask_keyboard,
    get_registry,
    render_ask_text,
)

logger = structlog.get_logger()


async def handle_interactive_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Dispatch on the callback_data prefix."""
    cb = update.callback_query
    if cb is None or cb.data is None:
        return
    data = cb.data
    if data.startswith("ans:"):
        await _handle_ask_callback(update, context, data)
    elif data.startswith("plan:"):
        await _handle_plan_callback(update, context, data)
    # voice / quick added in later commits


# --------------------------------------------------------------- ask handler

async def _handle_ask_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
) -> None:
    """Process a tap on an AskUserQuestion option button.

    callback_data formats:
      ans:<prompt_id>:<qidx>:<oidx>     single tap
      ans:<prompt_id>:<qidx>:done       multi-select submit
    """
    cb = update.callback_query
    if cb is None:
        return

    parts = data.split(":")
    if len(parts) != 4:
        await cb.answer("malformed callback")
        return
    _, prompt_id, qidx_s, action = parts

    registry = get_registry(context)
    prompt = await registry.get(prompt_id)
    if prompt is None or prompt.future.done():
        await cb.answer("This question already answered.", show_alert=False)
        return
    if cb.from_user is None or cb.from_user.id != prompt.user_id:
        await cb.answer("Not your prompt.", show_alert=False)
        return

    try:
        qidx = int(qidx_s)
    except ValueError:
        await cb.answer("malformed callback")
        return

    questions: List[Dict[str, Any]] = prompt.tool_input.get("questions") or []
    if qidx >= len(questions):
        await cb.answer("question out of range")
        return
    q = questions[qidx]
    multi = bool(q.get("multiSelect"))

    # Collect-or-submit logic.
    if action == "done":
        if not multi:
            await cb.answer("not a multi-select")
            return
        # Record the multi-select answer for this question and advance.
        sel = prompt.selections.get(qidx)
        chosen = sel.selected_indices if sel else []
        await _record_answer(prompt, qidx, chosen, multi=True)
        await cb.answer("recorded")
        await _advance_or_finish(prompt, context)
        return

    # Single-tap: it's either toggle (multi-select) or submit (single-select).
    try:
        oidx = int(action)
    except ValueError:
        await cb.answer("malformed callback")
        return

    if multi:
        sel = prompt.selections.setdefault(qidx, _Selection(question_index=qidx))
        if oidx in sel.selected_indices:
            sel.selected_indices.remove(oidx)
        else:
            sel.selected_indices.append(oidx)
        await cb.answer("toggled")
        # Re-render with the updated selection state.
        await _refresh_prompt_message(prompt, context)
    else:
        await _record_answer(prompt, qidx, [oidx], multi=False)
        await cb.answer("recorded")
        await _advance_or_finish(prompt, context)


async def _record_answer(
    prompt: PendingPrompt, qidx: int, chosen: List[int], multi: bool
) -> None:
    """Append to prompt.tool_input['answers'] in the SDK's expected shape."""
    questions: List[Dict[str, Any]] = prompt.tool_input.get("questions") or []
    if qidx >= len(questions):
        return
    q = questions[qidx]
    options = q.get("options") or []
    answers = prompt.tool_input.setdefault("answers", {})
    header = q.get("header") or f"q{qidx}"

    if multi:
        labels = [options[i].get("label", "") for i in chosen if i < len(options)]
        answers[header] = labels
    else:
        idx = chosen[0] if chosen else -1
        answers[header] = options[idx].get("label", "") if 0 <= idx < len(options) else ""


async def _advance_or_finish(
    prompt: PendingPrompt, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Either re-render for the next question or resolve the future."""
    questions: List[Dict[str, Any]] = prompt.tool_input.get("questions") or []
    answers: Dict[str, Any] = prompt.tool_input.get("answers") or {}
    # Find the next question whose header isn't in answers.
    next_idx: Optional[int] = None
    for i, q in enumerate(questions):
        header = q.get("header") or f"q{i}"
        if header not in answers:
            next_idx = i
            break

    if next_idx is None:
        # All done — resolve the future with the merged tool_input.
        await _finalize_prompt(prompt, context, answers)
        return

    # Re-render the prompt message with the next question.
    await _refresh_prompt_message(prompt, context, qidx_override=next_idx)


async def _refresh_prompt_message(
    prompt: PendingPrompt,
    context: ContextTypes.DEFAULT_TYPE,
    qidx_override: Optional[int] = None,
) -> None:
    """Edit the bot message in place to show updated keyboard / text."""
    questions = prompt.tool_input.get("questions") or []
    if qidx_override is not None:
        qidx = qidx_override
    else:
        # Re-render the same question (used for multi-select toggle).
        qidx = next(
            (i for i, q in enumerate(questions) if (q.get("header") or f"q{i}") not in
             (prompt.tool_input.get("answers") or {})),
            0,
        )
    text = render_ask_text(questions, qidx, prompt.selections)
    keyboard = build_ask_keyboard(prompt.prompt_id, questions, prompt.selections)
    if prompt.prompt_message is not None:
        try:
            await prompt.prompt_message.edit_text(
                text, parse_mode="HTML", reply_markup=keyboard
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to refresh prompt message", error=str(e))


async def _finalize_prompt(
    prompt: PendingPrompt,
    context: ContextTypes.DEFAULT_TYPE,
    answers: Dict[str, Any],
) -> None:
    """Resolve the future, edit the bot message to show the recorded answer."""
    registry = get_registry(context)
    await registry.pop(prompt.prompt_id)
    if not prompt.future.done():
        prompt.future.set_result({"answers": answers})
    if prompt.prompt_message is not None:
        try:
            summary = "\n".join(f"<b>{k}</b>: {_summary(v)}" for k, v in answers.items())
            await prompt.prompt_message.edit_text(
                f"✓ <b>Answered</b>\n\n{summary}",
                parse_mode="HTML",
                reply_markup=None,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to finalize prompt message", error=str(e))


def _summary(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or "(none)"
    return str(value)


# --------------------------------------------------------- plan handler

# In-memory: maps user_id -> prompt_id awaiting modify-text reply. The next
# free-text message from that user is captured as plan feedback rather than a
# new turn. Cleared on tap.
_PENDING_MODIFY: Dict[int, str] = {}


def get_pending_modify_prompt_id(user_id: int) -> Optional[str]:
    """If the user has a plan-Modify reply pending, return the prompt id."""
    return _PENDING_MODIFY.get(user_id)


def clear_pending_modify(user_id: int) -> None:
    _PENDING_MODIFY.pop(user_id, None)


async def deliver_plan_modify(
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    feedback_text: str,
) -> bool:
    """Route a free-text reply from the user to a pending plan-modify prompt.

    Returns True if a pending prompt was found and resolved (caller should
    suppress the normal 'new turn' processing for this message).
    """
    prompt_id = _PENDING_MODIFY.pop(user_id, None)
    if prompt_id is None:
        return False
    registry = get_registry(context)
    prompt = await registry.pop(prompt_id)
    if prompt is None or prompt.future.done():
        return False
    # Tell Claude: keep the plan, but here is the user's feedback. The SDK
    # will receive {plan: "<feedback>"} as updated_input and Claude reads it.
    if not prompt.future.done():
        prompt.future.set_result({"plan": feedback_text})
    if prompt.prompt_message is not None:
        try:
            await prompt.prompt_message.edit_text(
                "✏ <b>Plan modified</b>\n\n"
                f"<i>Your feedback:</i>\n<blockquote>{_short(feedback_text)}</blockquote>",
                parse_mode="HTML",
                reply_markup=None,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to edit plan-modify message", error=str(e))
    return True


def _short(s: str, n: int = 600) -> str:
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


async def _handle_plan_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
) -> None:
    """Process Approve / Modify / Reject taps for plan mode."""
    cb = update.callback_query
    if cb is None:
        return
    parts = data.split(":")
    if len(parts) != 3:
        await cb.answer("malformed callback")
        return
    _, prompt_id, action = parts

    registry = get_registry(context)
    prompt = await registry.get(prompt_id)
    if prompt is None or prompt.future.done():
        await cb.answer("This plan already answered.")
        return
    if cb.from_user is None or cb.from_user.id != prompt.user_id:
        await cb.answer("Not your prompt.")
        return

    if action == "approve":
        await registry.pop(prompt_id)
        if not prompt.future.done():
            # Pass through the original tool_input — Claude proceeds.
            prompt.future.set_result(dict(prompt.tool_input))
        await cb.answer("Approved")
        if prompt.prompt_message is not None:
            try:
                await prompt.prompt_message.edit_reply_markup(reply_markup=None)
                await prompt.prompt_message.reply_text(
                    "✓ <b>Plan approved</b> — proceeding.",
                    parse_mode="HTML",
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to ack plan approve", error=str(e))
        return

    if action == "reject":
        await registry.pop(prompt_id)
        if not prompt.future.done():
            prompt.future.set_result(None)  # → PermissionResultDeny
        await cb.answer("Rejected")
        if prompt.prompt_message is not None:
            try:
                await prompt.prompt_message.edit_reply_markup(reply_markup=None)
                await prompt.prompt_message.reply_text(
                    "✗ <b>Plan rejected</b>.",
                    parse_mode="HTML",
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to ack plan reject", error=str(e))
        return

    if action == "modify":
        # Mark this user as awaiting a free-text reply that becomes the
        # plan-modify feedback. The future stays unresolved until the reply
        # arrives (or the prompt times out).
        _PENDING_MODIFY[prompt.user_id] = prompt_id
        await cb.answer("Reply with your feedback")
        if prompt.prompt_message is not None:
            try:
                await prompt.prompt_message.edit_reply_markup(reply_markup=None)
                await prompt.prompt_message.reply_text(
                    "✏ <b>Modify plan</b> — reply to this message with your feedback.\n"
                    "<i>(or send /cancel to abort)</i>",
                    parse_mode="HTML",
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to ack plan modify", error=str(e))
        return

    await cb.answer("unknown action")
