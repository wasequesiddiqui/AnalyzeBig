import logging
from pathlib import Path

LOG_FILE_NAME = "exception_log.txt"
LOG_FILE_PATH = Path(__file__).resolve().parent / LOG_FILE_NAME


def _get_logger():
    logger = logging.getLogger("EquityFinderExceptionLogger")
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    return logger


def log_exception(exc: Exception, context: str = None):
    """Log an exception to the shared exception_log.txt file.

    Args:
        exc (Exception): The exception object.
        context (str, optional): A short context string describing where the exception occurred.
    """
    logger = _get_logger()
    message = (
        f"[{context}] {type(exc).__name__}: {exc}" if context else f"{type(exc).__name__}: {exc}"
    )
    logger.error(message, exc_info=True)
    return LOG_FILE_PATH
