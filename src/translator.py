from pathlib import Path
from typing import List, Optional
import ctranslate2
import transformers
from src.utils import logger, get_device
from src.config import config

class Translator:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.get("model.ct2_model_path")
        self.device = get_device() if config.get("model.device") == "auto" else config.get("model.device")
        self.compute_type = config.get("model.compute_type", "int8")
        
        self.translator = None
        self.tokenizer = None
        
        logger.info(f"Initializing Translator on {self.device} with {self.compute_type} quantization")
