from pathlib import Path
import logging


LOG_FILE_PATH = "lstm_logs.log"
LOGGER_NAME = "lstm_logger"


def get_logger() -> logging.Logger:
    log_file = Path(LOG_FILE_PATH)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    return logger


logger = get_logger()
