from loguru import logger
import os
import datetime

class CustomLogger:
    def __init__(self, log_dir="logs", log_file_prefix="app"):
        """
        Custom logger using loguru that saves logs to a file and returns the log file path.
        
        :param log_dir: Directory where log files will be stored.
        :param log_file_prefix: Prefix for log filenames.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists

        # Generate log file name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(log_dir, f"{log_file_prefix}_{timestamp}.log")

        # Configure loguru logger
        logger.add(self.log_file, rotation="10MB", retention="7 days", level="DEBUG")

    def log(self, level, message):
        """
        Logs a message at a given level and returns the log file path.
        
        :param level: Logging level ('info', 'debug', 'warning', 'error', 'critical').
        :param message: Message to log.
        :return: Log file path.
        """
        log_methods = {
            "info": logger.info,
            "debug": logger.debug,
            "warning": logger.warning,
            "error": logger.error,
            "critical": logger.critical
        }

        if level not in log_methods:
            raise ValueError("Invalid log level. Choose from: info, debug, warning, error, critical.")

        log_methods[level](message)  # Log message
        return self.log_file  # Return log file path



