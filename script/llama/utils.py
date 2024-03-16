import os

from logging import getLogger, Formatter, StreamHandler, DEBUG, INFO


def _set_handler(logger, handler, verbose: bool):
    """
    Prep handler
    """
    if verbose:
        handler.setLevel(DEBUG)
    else:
        handler.setLevel(INFO)
    formatter = Formatter(
        "%(asctime)s %(name)s:%(lineno)s [%(levelname)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_module_logger(verbose: bool = False, level=DEBUG):
    """
    Create logger
    """
    logger = getLogger(__name__)
    logger = _set_handler(logger, StreamHandler(), False)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def makedirs_recursive(path):
    if not os.path.exists(path):
        os.makedirs(path)
