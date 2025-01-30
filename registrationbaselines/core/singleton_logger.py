
import logging
from pathlib import Path

MY_LOG_LEVEL = 25


class SingletonLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        log_file_path: Path,
        level_name: str,
        level=MY_LOG_LEVEL
    ) -> None:
        if not self._initialized:  # Ensure initialization only happens once
            self._initialized = True

            logging.addLevelName(MY_LOG_LEVEL, level_name)

            def my_level(self, message, *args, **kwargs):
                if self.isEnabledFor(MY_LOG_LEVEL):
                    self._log(MY_LOG_LEVEL, message, args, **kwargs)

            logging.Logger.my_level = my_level

            logging.basicConfig(
                filename=log_file_path,
                level=level,
                format="%(asctime)s - %(levelname)s - %(message)s"
            )
            self.logger = logging.getLogger("singleton_logger")

    @staticmethod
    def get_logger():
        if not SingletonLogger._instance:
            raise Exception(
                "Logger not initialized. Create an instance first.")
        return SingletonLogger._instance.logger
