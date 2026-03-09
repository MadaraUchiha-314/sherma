import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Convention: each module calls ``logger = get_logger(__name__)``.
    """
    return logging.getLogger(name)
