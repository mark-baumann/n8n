import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOGS_DIR = "logs"

def setup_logging():
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler that rotates daily
    handler = TimedRotatingFileHandler(
        os.path.join(LOGS_DIR, "app.log"),
        when="midnight",
        backupCount=7
    )
    handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)
