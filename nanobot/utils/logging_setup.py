"""structlog configuration for nanobot."""

import logging
import sys

import structlog


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            _brace_format_processor,
            structlog.stdlib.ExtraAdder(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _brace_format_processor(_, __, event_dict):
    args = event_dict.pop("positional_args", ())
    if args:
        try:
            event_dict["event"] = str(event_dict["event"]).format(*args)
        except Exception:
            event_dict["event"] = str(event_dict["event"]) + " " + " ".join(str(a) for a in args)
    return event_dict
