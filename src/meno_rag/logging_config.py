import logging
import sys
from typing import Any

import structlog

# Whether log lines may carry short excerpts of message content (the model's answer, the
# rewritten search queries). Useful for debugging a bad answer; also personal data, so it
# is a deliberate switch rather than a hardcoded choice — see `preview_field`.
_CONTENT_PREVIEWS = True


def configure_logging(level: str = "INFO", *, content_previews: bool = True) -> None:
    global _CONTENT_PREVIEWS
    _CONTENT_PREVIEWS = content_previews
    _configure(level)


def preview_field(name: str, text: str, limit: int) -> dict[str, Any]:
    """A log field holding a content excerpt, or nothing at all when previews are off.

    Splat it into the log call (``**preview_field("content_preview", answer, 200)``) so a
    disabled preview leaves no key behind rather than logging an empty value.

    Excerpts of a dialogue are personal data: they are lawful to log for diagnosing errors
    and keeping the service running, but only while the log has an approved retention
    period, restricted access, and the processing is described in the published Policy.
    ``LOG_CONTENT_PREVIEWS=false`` turns them off wherever that does not hold.
    """
    if not _CONTENT_PREVIEWS or not text:
        return {}
    return {name: text[:limit]}


def _configure(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
