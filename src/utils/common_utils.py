import functools
import logging
import os
from datetime import datetime
import pathlib
from typing import Union


# Set up logging with environment variable
log_level_str = os.environ.get("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def time_execution(func):
    """
    Decorator to time the execution of a function.
    Prints execution time but returns only the original result.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        elapsed_time = datetime.now() - start_time
        logger.info(f"{func.__name__} completed in {elapsed_time}")
        return result

    return wrapper


def get_logger():
    return logger


def path_exists(path: Union[pathlib.Path, str]) -> bool:
    """
    Check if a path exists. Accepts both pathlib.Path and string paths.

    Args:
        path: pathlib.Path or string representing the path to check.

    Returns:
        True if the path exists, False otherwise.
    """
    path_obj = pathlib.Path(path) if isinstance(path, str) else path
    return path_obj.exists()
