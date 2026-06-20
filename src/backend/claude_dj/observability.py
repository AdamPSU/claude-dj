from __future__ import annotations

from collections.abc import Awaitable, Callable
import logging
import os
from typing import Any, TypeVar

import sentry_sdk
from dotenv import load_dotenv
from sentry_sdk.integrations.logging import LoggingIntegration

_initialized = False
T = TypeVar("T")


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


async def observe_async(
    operation: str,
    *,
    op: str,
    data: dict[str, Any] | None,
    callback: Callable[[], Awaitable[T]],
) -> T:
    attributes = _span_data(data or {})
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
