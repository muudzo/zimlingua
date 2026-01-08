import logging
import sys
import torch
from pathlib import Path

def setup_logger(name: str = "zimlingua", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with standard formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def get_device() -> str:
    """
    Detects the best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return "cuda"
    # elif torch.backends.mps.is_available(): # MPS support for Mac M1/M2 usually requires verification for CTranslate2
    #     return "cpu" # Safest default for now, can enable mps if supported by backend
    else:
        return "cpu"

logger = setup_logger()
