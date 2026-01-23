import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from datetime import datetime


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> structlog.BoundLogger:
    """
    Setup structured logger with console and file handlers.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    
    Returns:
        Configured structlog logger
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(colors=True)
    ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger(name)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s [%(levelname)8s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logging.getLogger().addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)
