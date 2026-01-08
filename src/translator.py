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
        self.load_model()

    def load_model(self):
        """
        Loads the CTranslate2 model and the HuggingFace tokenizer.
        """
        if not Path(self.model_path).exists():
            logger.error(f"Model path {self.model_path} does not exist. Run scripts/download_model.py and scripts/convert_model.py first.")
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        try:
            self.translator = ctranslate2.Translator(
                self.model_path,
                device=self.device,
                compute_type=self.compute_type
            )
            
            # Using NLLB tokenizer from HF
            # We assume the tokenizer vocab/config is inside the converted model dir or we fetch from original HF name
            # CTranslate2 models usually contain the SP model but for NLLB we need the HF tokenizer for correct pre-processing
            model_name = config.get("model.name", "facebook/nllb-200-distilled-600M")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            
            logger.info("Model and Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model architecture: {e}")
            raise
