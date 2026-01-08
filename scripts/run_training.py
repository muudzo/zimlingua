import argparse
from src.trainer import ZimTrainer
from src.data_loader import DataLoader
from src.split_utils import split_data
from src.utils import logger
from datasets import Dataset

def main(data_file: str, source_lang: str, target_lang: str, epochs: int, batch_size: int, output_dir: str):
    logger.info(f"Starting training run on {data_file}")
    
    # 1. Load Data
    loader = DataLoader() # Assuming default data dir
    # Basic detection of file type
    if data_file.endswith(".csv"):
        raw_data = loader.load_csv(data_file, source_col="source", target_col="target")
    else:
        # Assuming parallel text files? For now just support CSV in this entry point or fail
        logger.error("Only CSV supported for this script currently. Use custom loading for TXT.")
        return

    # 2. Preprocessing
    # Normalize
    processed_data = []
    for item in raw_data:
        processed_data.append({
            "source": loader.normalize_text(item["source"]),
            "target": loader.normalize_text(item["target"])
        })
    
    # Deduplicate
    processed_data = loader.deduplicate(processed_data)
    
    # Split
    train_data, val_data, _ = split_data(processed_data, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0)
    
    # Convert to HF Datasets
    # NLLB expects specific input format, usually handled by tokenization in collator or map.
    # Here we typically need to tokenize before passing to trainer or use a map function.
    # For simplicity in this skeleton, we'll let the Collator handle on-the-fly if configured,
    # BUT standard Seq2SeqTrainer expects tokenized inputs 'input_ids', 'labels'.
    
    trainer = ZimTrainer(output_dir=output_dir)
    trainer.load_base_model()
    trainer.setup_lora()
    
    def tokenize_function(examples):
        inputs = examples["source"]
        targets = examples["target"]
        model_inputs = trainer.tokenizer(inputs, max_length=128, truncation=True)
        # Setup the tokenizer for targets
        with trainer.tokenizer.as_target_tokenizer():
            labels = trainer.tokenizer(targets, max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    hf_train = Dataset.from_list(train_data)
    hf_val = Dataset.from_list(val_data)
    
    tokenized_train = hf_train.map(tokenize_function, batched=True)
    tokenized_val = hf_val.map(tokenize_function, batched=True)
    
    # 3. Train
    trainer.train(tokenized_train, tokenized_val, batch_size=batch_size, epochs=epochs)
    
    # 4. Save
    trainer.save_model("final_adapter")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NLLB on custom data")
    parser.add_argument("--data", type=str, required=True, help="Path to data file (CSV) relative to data/")
    parser.add_argument("--src_lang", type=str, default="eng_Latn", help="Source language code")
    parser.add_argument("--tgt_lang", type=str, default="sna_Latn", help="Target language code")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints", help="Output directory")
    
    args = parser.parse_args()
    main(args.data, args.src_lang, args.tgt_lang, args.epochs, args.batch_size, args.output_dir)
