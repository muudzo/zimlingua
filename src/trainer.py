import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from typing import Optional, Dict
from src.utils import logger, get_device
from src.config import config

class ZimTrainer:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", output_dir: str = "models/checkpoints"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = get_device()
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        logger.info(f"Initializing ZimTrainer with model {model_name}")

    def load_base_model(self):
        """
        Loads the base model and tokenizer for training.
        """
        logger.info("Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Move to device if needed, though HF Trainer handles this usually
        if self.device == "cuda":
            self.model.to("cuda")
