import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
import sys

# Add src to path to import utils if needed, or just use standalone
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import logger

def download_model(model_name: str, cache_dir: str):
    logger.info(f"Downloading model {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        logger.info(f"Successfully downloaded {model_name} to {cache_dir}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NLLB model from HuggingFace")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-distilled-600M", help="Model ID")
    parser.add_argument("--dir", type=str, default="models/hf_cache", help="Directory to save model")
    
    args = parser.parse_args()
    download_model(args.model, args.dir)
