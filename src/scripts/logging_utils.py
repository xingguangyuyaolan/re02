"""Centralized logging setup for training and environment runtime."""

from datetime import datetime
import logging
import os
import sys
import threading


def _build_formatter():
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_project_logging(output_root: str, run_name: str, console_level: int = logging.INFO):
    """Configure root logger and return generated log file paths.

    Creates two files under artifacts: a full runtime log and an error-only log.
    """
    run_dir = os.path.join(output_root, run_name)
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    runtime_log_path = os.path.join(logs_dir, "runtime.log")
    error_log_path = os.path.join(logs_dir, "errors.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Reset handlers so repeated invocations (for example during notebook/debug runs)
    # do not duplicate the same messages.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = _build_formatter()

    file_handler = logging.FileHandler(runtime_log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    error_handler = logging.FileHandler(error_log_path, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    def _log_excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            root_logger.info("KeyboardInterrupt received, shutting down.")
            return
        root_logger.exception(
            "Unhandled exception captured by global excepthook",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = _log_excepthook

    if hasattr(threading, "excepthook"):
        def _thread_excepthook(args):
            root_logger.exception(
                "Unhandled thread exception in '%s'",
                getattr(args.thread, "name", "unknown"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = _thread_excepthook

    root_logger.info(
        "Logging initialized | run_name=%s | runtime_log=%s | error_log=%s | started_at=%s",
        run_name,
        runtime_log_path,
        error_log_path,
        datetime.now().isoformat(timespec="seconds"),
    )
    return {
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "runtime_log_path": runtime_log_path,
        "error_log_path": error_log_path,
    }
