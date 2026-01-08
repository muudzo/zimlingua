import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import logger

def merge_lora(base_model_name: str, adapter_path: str, output_dir: str):
    logger.info(f"Merging LoRA adapter from {adapter_path} into {base_model_name}...")
    
    # Load base
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge
    model = model.merge_and_unload()
    
    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Merged model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base_model", type=str, default="facebook/nllb-200-distilled-600M", help="Base model ID")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model")
    
    args = parser.parse_args()
    merge_lora(args.base_model, args.adapter, args.output)
