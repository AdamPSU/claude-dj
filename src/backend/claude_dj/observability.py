from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
from typing import Any, TypeVar
from uuid import uuid4

import sentry_sdk
from dotenv import load_dotenv
from sentry_sdk.integrations.logging import LoggingIntegration

_initialized = False
T = TypeVar("T")


@dataclass(frozen=True)
class RunContext:
    session_id: str
    run_id: str
    run_type: str


_current_run: ContextVar[RunContext | None] = ContextVar("claude_dj_current_run", default=None)
_tool_sequence: ContextVar[int] = ContextVar("claude_dj_tool_sequence", default=0)


def init_sentry() -> None:
    global _initialized

    if _initialized:
        return

    load_dotenv()

    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        environment=os.environ.get("SENTRY_ENVIRONMENT"),
        release=os.environ.get("SENTRY_RELEASE"),
        send_default_pii=False,
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
        enable_logs=True,
        integrations=[
            LoggingIntegration(
                sentry_logs_level=logging.INFO,
                level=logging.INFO,
                event_level=logging.ERROR,
            ),
        ],
        in_app_include=["claude_dj"],
        ignore_errors=[KeyboardInterrupt],
    )
    sentry_sdk.get_global_scope().set_tag("service", "claude_dj_backend")
    _initialized = True


def add_breadcrumb(message: str, *, category: str, data: dict[str, Any] | None = None, level: str = "info") -> None:
    sentry_sdk.add_breadcrumb(
        category=category,
        message=message,
        level=level,
        data=_span_data(data or {}),
    )


def capture_swallowed_exception(
    exc: BaseException,
    *,
    operation: str,
    data: dict[str, Any] | None = None,
) -> None:
    with sentry_sdk.new_scope() as scope:
        scope.set_tag("service", "claude_dj_backend")
        scope.set_tag("operation", operation)
        scope.set_context("claude_dj", _span_data(data or {}))
        sentry_sdk.capture_exception(exc)


def capture_warning(message: str, *, operation: str, data: dict[str, Any] | None = None) -> None:
    with sentry_sdk.new_scope() as scope:
        scope.set_tag("service", "claude_dj_backend")
        scope.set_tag("operation", operation)
        scope.set_context("claude_dj", _span_data(data or {}))
        sentry_sdk.capture_message(message, level="warning")


async def observe_run(
    run_type: str,
    *,
    session_id: str,
    data: dict[str, Any] | None,
    callback: Callable[[], Awaitable[T]],
) -> T:
    run_id = uuid4().hex
    context = RunContext(session_id=session_id, run_id=run_id, run_type=run_type)
    token_run = _current_run.set(context)
    token_tool_sequence = _tool_sequence.set(0)
    attributes = _span_data(
        {
            "claude_dj.session_id": session_id,
            "claude_dj.run_id": run_id,
            "claude_dj.run_type": run_type,
            **(data or {}),
        }
    )
    add_breadcrumb(f"claude_dj.run.{run_type} started", category="claude_dj.run", data=attributes)
    try:
        with sentry_sdk.start_transaction(op="claude_dj.run", name=f"claude_dj.run.{run_type}") as transaction:
            for key, value in attributes.items():
                transaction.set_data(key, value)
            try:
                result = await callback()
            except Exception as exc:
                transaction.set_data("failed", True)
                transaction.set_data("tool_count", _tool_sequence.get())
                capture_swallowed_exception(exc, operation=f"claude_dj.run.{run_type}", data=attributes)
                raise
            transaction.set_data("failed", False)
            transaction.set_data("tool_count", _tool_sequence.get())
            add_breadcrumb(f"claude_dj.run.{run_type} completed", category="claude_dj.run", data=attributes)
            return result
    finally:
        _tool_sequence.reset(token_tool_sequence)
        _current_run.reset(token_run)


async def observe_async(
    operation: str,
    *,
    op: str,
    data: dict[str, Any] | None,
    callback: Callable[[], Awaitable[T]],
) -> T:
    attributes = _span_data(data or {})
    run_context = _current_run.get()
    if run_context is not None:
        attributes["claude_dj.session_id"] = run_context.session_id
        attributes["claude_dj.run_id"] = run_context.run_id
        attributes["claude_dj.run_type"] = run_context.run_type
        if op == "mcp.tool":
            tool_index = _tool_sequence.get() + 1
            _tool_sequence.set(tool_index)
            attributes["claude_dj.tool_index"] = tool_index
    if op == "mcp.tool" and "tool" in attributes:
        attributes["mcp.tool.name"] = str(attributes["tool"])
    add_breadcrumb(f"{operation} started", category="claude_dj", data=attributes)
    with sentry_sdk.start_span(op=op, name=operation) as span:
        for key, value in attributes.items():
            span.set_data(key, value)
        try:
            result = await callback()
        except Exception as exc:
            span.set_data("failed", True)
            capture_swallowed_exception(exc, operation=operation, data=attributes)
            raise
        span.set_data("failed", False)
        add_breadcrumb(f"{operation} completed", category="claude_dj", data=attributes)
        return result


def _span_data(data: dict[str, Any]) -> dict[str, str | int | float | bool | list[str]]:
    safe: dict[str, str | int | float | bool | list[str]] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            safe[key] = value
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            safe[key] = value
            continue
        safe[key] = str(value)
    return safe
