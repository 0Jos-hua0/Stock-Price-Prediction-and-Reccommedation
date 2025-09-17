"""Logging configuration and utilities for the application."""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime

from app.core.config import settings

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_record: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in ('args', 'asctime', 'created', 'exc_info', 'exc_text', 
                          'levelname', 'levelno', 'message', 'msg', 'name', 'pathname',
                          'process', 'processName', 'relativeCreated', 'stack_info', 'thread',
                          'threadName', 'extra'):
                log_record[key] = value
        
        # Add any extra attributes from record.extra
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        
        return json.dumps(log_record, ensure_ascii=False)

def setup_logging() -> None:
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Set log level from settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # File handler with rotation
    log_file = log_dir / 'app.log'
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    
    # Formatter
    if settings.ENVIRONMENT == 'production':
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Set log level for third-party libraries
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger.info("Logging configured")

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: The name of the logger. If None, returns the root logger.
        
    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)

# Initialize logging when module is imported
setup_logging()
logger = get_logger(__name__)
