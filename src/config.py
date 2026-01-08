import yaml
from pathlib import Path
from typing import Any, Dict
from src.utils import logger

class Config:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """
        Loads configuration from config.yaml.
        """
        config_path = Path("config.yaml")
        if not config_path.exists():
            logger.error(f"Config file not found at {config_path.absolute()}")
            raise FileNotFoundError("config.yaml not found")
        
        try:
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value by key (supports nested keys with dot notation).
        """
        keys = key.split(".")
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default

    @property
    def all(self) -> Dict[str, Any]:
        return self._config

# Global instance
config = Config()
