"""Logger configuration for the probing module.

Provides structured logging setup with both file and console outputs.
"""

import sys
from collections.abc import Callable
from pathlib import Path

from loguru import logger


class ProbeLogger:
    """Handles logging configuration for the probing module."""

    DEFAULT_FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    CONSOLE_FORMAT = (
        "<level>{level: <8}</level> | "
        "<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    )

    def __init__(
        self,
        log_dir: str | Path | None = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        rotation: str = "100 MB",
        retention: str = "1 week",
        tqdm_compatible: bool = True
    ):
        """Initialize logger configuration.

        Args:
            log_dir: Directory to store log files. If None, uses current directory
            console_level: Logging level for console output
            file_level: Logging level for file output
            rotation: When to rotate log files (size or time)
            retention: How long to keep log files
            tqdm_compatible: Whether to make console output compatible with tqdm
        """
        self.log_dir = Path(log_dir) if log_dir else Path.cwd()
        self.console_level = console_level
        self.file_level = file_level
        self.rotation = rotation
        self.retention = retention
        self.tqdm_compatible = tqdm_compatible

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def setup(self, name: str = "shrunkiq_probe") -> None:
        """Set up logging configuration.

        Args:
            name: Base name for log files
        """
        # Remove any existing handlers
        logger.remove()

        # Add file handler
        log_file = self.log_dir / f"{name}.log"
        logger.add(
            log_file,
            level=self.file_level,
            format=self.DEFAULT_FORMAT,
            rotation=self.rotation,
            retention=self.retention,
            enqueue=True  # Thread-safe logging
        )

        # Add console handler
        console_sink: Callable
        if self.tqdm_compatible:
            # Use tqdm.write for progress bar compatibility
            from tqdm import tqdm
            console_sink = lambda msg: tqdm.write(msg, end="") # noqa: E731
        else:
            console_sink = sys.stderr

        logger.add(
            console_sink,
            level=self.console_level,
            format=self.CONSOLE_FORMAT,
            colorize=True,
            enqueue=True
        )

    @staticmethod
    def get_logger():
        """Get configured logger instance."""
        return logger

    def set_console_level(self, level: str) -> None:
        """Dynamically change console output level."""
        self.console_level = level
        self.setup()  # Reinitialize with new level

    def set_file_level(self, level: str) -> None:
        """Dynamically change file output level."""
        self.file_level = level
        self.setup()  # Reinitialize with new level

# Create default logger instance
probe_logger = ProbeLogger()
