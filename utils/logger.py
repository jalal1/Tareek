import logging
import sys
from pathlib import Path
from datetime import datetime

# Cross-platform symbols that work on all systems
CHECK_MARK = '[OK]'
CROSS_MARK = '[X]'

# Get the project root directory (assuming logger.py is in utils folder)
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'

# Global variable to store the current log file path
_current_log_file = None
_current_experiment_dir = None


def get_timestamped_log_filename(prefix: str = 'plan_generation') -> str:
    """Generate a timestamped log filename."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{prefix}_{timestamp}.log'


def get_experiment_dir_name() -> str:
    """Generate a timestamped experiment directory name."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'experiment_{timestamp}'


def create_experiment_dir() -> Path:
    """Create a new experiment directory and return its path."""
    global _current_experiment_dir

    # Create experiments directory if it doesn't exist
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamped experiment directory
    exp_dir = EXPERIMENTS_DIR / get_experiment_dir_name()
    exp_dir.mkdir(parents=True, exist_ok=True)

    _current_experiment_dir = exp_dir
    return exp_dir


def get_current_experiment_dir() -> Path:
    """Get the current experiment directory, creating one if it doesn't exist."""
    global _current_experiment_dir
    if _current_experiment_dir is None:
        _current_experiment_dir = create_experiment_dir()
    return _current_experiment_dir


def setup_logger(name: str = None, log_file: str = None, level=logging.INFO, use_experiment_dir: bool = False, log_to_file: bool = True, log_to_console: bool = True) -> logging.Logger:
    """Set up logger with consistent configuration

    Args:
        name (str): Logger name (usually __name__ from the calling module)
        log_file (str): Log filename (will be placed in project's logs directory or experiment directory).
                       If None, generates timestamped filename.
        level: Logging level (default: INFO)
        use_experiment_dir (bool): If True, log to experiment directory instead of logs directory
        log_to_file (bool): If False, don't write to file (reduces file size for large runs)
        log_to_console (bool): If False, don't write to console
    """
    global _current_log_file

    # Create logger with given name
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)

    # Only add handlers if they haven't been added yet
    # This prevents worker processes from creating duplicate log files
    if not getattr(logging, '_is_logger_configured', False):
        # Determine log directory
        if use_experiment_dir:
            log_dir = get_current_experiment_dir()
        else:
            log_dir = LOG_DIR

        # Ensure log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log filename
        if log_file is None:
            log_file = get_timestamped_log_filename()

        log_path = log_dir / log_file
        _current_log_file = str(log_path)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Console handler (only if log_to_console is True)
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(level)
            root_logger.addHandler(console_handler)

        # File handler (only if log_to_file is True)
        if log_to_file:
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')  # UTF-8 encoding for cross-platform support
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)

        setattr(logging, '_is_logger_configured', True)

        # Log the log file location
        logger.info(f"Logging to: {log_path}")

    return logger


def get_current_log_file() -> str:
    """Get the path to the current log file."""
    return _current_log_file or str(LOG_DIR / 'log.log')


def reconfigure_logger_to_experiment_dir(experiment_dir: Path, log_prefix: str = 'experiment') -> Path:
    """
    Reconfigure the existing logger to write to the experiment directory instead.
    This moves the log file from logs/ to the experiment directory.

    Args:
        experiment_dir: Path to the experiment directory
        log_prefix: Prefix for the log filename (default: 'experiment')

    Returns:
        Path to the new log file in the experiment directory
    """
    global _current_log_file
    global _current_experiment_dir

    # If already configured for this experiment directory, return existing path
    if _current_log_file and _current_experiment_dir == experiment_dir:
        return Path(_current_log_file)

    # Update current experiment directory
    _current_experiment_dir = experiment_dir

    # Generate new log file path in experiment directory
    log_filename = get_timestamped_log_filename(prefix=log_prefix)
    new_log_path = experiment_dir / log_filename

    # Get root logger
    root_logger = logging.getLogger()

    # Find and remove existing file handlers
    handlers_to_remove = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
    for handler in handlers_to_remove:
        handler.close()
        root_logger.removeHandler(handler)

    # Create new file handler in experiment directory
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Copy content from old log file before creating new handler
    old_log_file = _current_log_file
    old_log_content = None
    if old_log_file and Path(old_log_file).exists():
        try:
            old_log_content = Path(old_log_file).read_text(encoding='utf-8')
        except Exception:
            pass

    # Write old content first, then open in append mode so new logs follow
    if old_log_content:
        new_log_path.write_text(old_log_content, encoding='utf-8')
        file_handler = logging.FileHandler(new_log_path, mode='a', encoding='utf-8')
    else:
        file_handler = logging.FileHandler(new_log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Update global log file path
    _current_log_file = str(new_log_path)

    # Log the transition
    logger = logging.getLogger(__name__)
    logger.info(f"Logging reconfigured from {old_log_file} to {new_log_path}")

    # Delete the old temporary log file now that its content has been preserved
    if old_log_file and Path(old_log_file).exists():
        try:
            Path(old_log_file).unlink()
            logger.info(f"Removed temporary log file: {old_log_file}")
        except Exception as e:
            logger.warning(f"Could not remove temporary log file {old_log_file}: {e}")

    return new_log_path