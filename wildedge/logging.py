import logging

logger = logging.getLogger("wildedge")


def enable_debug() -> None:
    """Set up a stderr handler and enable DEBUG level on the wildedge logger."""
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
