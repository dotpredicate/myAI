import logging
import sys

RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[36m",      # Cyan
    "INFO": "\033[32m",       # Green
    "WARNING": "\033[33m",    # Yellow
    "ERROR": "\033[31m",      # Red
    "CRITICAL": "\033[1;31m", # Bold Red
}
LEVEL_COLOR = "\033[37m"     # White (grey) for level name

class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color = COLORS.get(levelname, RESET)
        record.levelname = f"{LEVEL_COLOR}{levelname}{RESET}"
        formatted = super().format(record)
        record.levelname = levelname  # restore
        return f"{color}{formatted}{RESET}"

def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)

    # Apply the same colored format to uvicorn loggers
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.propagate = False

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
