import logging
import sys

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Create a custom logger
logger = logging.getLogger("TSLib")
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(LOG_FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler (optional, can be enabled if needed)
# fh = logging.FileHandler('tslib.log')
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logger.addHandler(fh)

def set_log_level(level):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

# Usage:
# from utils.logger import logger
# logger.info("message")
# logger.debug("debug message")
# set_log_level(logging.DEBUG)
