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

    def setup_lora(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        """
        Configures the model for Low-Rank Adaptation (LoRA) to enable efficient fine-tuning.
        """
        from peft import get_peft_model, LoraConfig, TaskType
        
        logger.info(f"Setting up LoRA (r={r}, alpha={lora_alpha}, dropout={lora_dropout})...")
        
        # Target modules for NLLB/M2M100 usually include query/key/value/output projections
        # 'q_proj', 'v_proj' are safe defaults for most transformers
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"] 
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
