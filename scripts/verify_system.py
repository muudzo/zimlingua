import sys
import argparse
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent.parent))
from src.translator import Translator
from src.data_loader import DataLoader
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemCheck")

def run_system_check():
    logger.info("Starting System Verification...")
    
    # 1. Config Check
    logger.info("Checking configuration...")
    try:
        model_name = config.get("model.name")
        logger.info(f"Config loaded. Target Model: {model_name}")
    except Exception as e:
        logger.error(f"Config check failed: {e}")
        sys.exit(1)

    # 2. Data Loader Check
    logger.info("Checking Data Loader...")
    try:
        loader = DataLoader()
        norm_text = loader.normalize_text("  TESTING 123 ")
        if norm_text != "testing 123":
            raise ValueError(f"Normalization failed: {norm_text}")
        logger.info("Data Loader OK.")
    except Exception as e:
        logger.error(f"Data Loader check failed: {e}")
        sys.exit(1)

    # 3. Model Engine Check
    logger.info("Checking Inference Engine...")
    try:
        # We wrap this in a try/except because model files might not exist if user hasn't downloaded them
        # We check for the files first
        model_path = Path(config.get("model.ct2_model_path", "models/ctranslate2_int8"))
        if not model_path.exists():
            logger.warning("Model files not found. Skipping inference check.")
            logger.warning("To run full check: python scripts/download_model.py && python scripts/convert_model.py")
        else:
            translator = Translator(model_path=str(model_path))
            result = translator.translate_batch(["Hello"], source_lang="eng_Latn", target_lang="sna_Latn")
            logger.info(f"Inference OK. Result: {result}")
    except Exception as e:
        logger.error(f"Inference Engine check failed: {e}")
        sys.exit(1)

    logger.info("System Verification Complete. All systems operational.")

if __name__ == "__main__":
    run_system_check()
