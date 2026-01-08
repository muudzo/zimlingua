import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from src.utils import logger
from src.config import config

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)

    def load_csv(self, filename: str, source_col: str, target_col: str) -> List[Dict[str, str]]:
        """
        Loads a CSV dataset and returns a list of source-target pairs.
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"{file_path} does not exist")

        try:
            df = pd.read_csv(file_path)
            if source_col not in df.columns or target_col not in df.columns:
                raise ValueError(f"Columns {source_col} and/or {target_col} missing in CSV")
            
            # Drop NaN values
            df = df.dropna(subset=[source_col, target_col])
            
            data = []
            for _, row in df.iterrows():
                data.append({
                    "source": str(row[source_col]),
                    "target": str(row[target_col])
                })
            
            logger.info(f"Loaded {len(data)} pairs from {filename}")
            return data
        except Exception as e:
            logger.error(f"Error loading CSV {filename}: {e}")
            raise

    def load_txt(self, source_file: str, target_file: str) -> List[Dict[str, str]]:
        """
        Loads parallel TXT files (line-by-line alignment).
        """
        src_path = self.data_dir / source_file
        tgt_path = self.data_dir / target_file

        if not src_path.exists() or not tgt_path.exists():
            raise FileNotFoundError("Source or target file not found")

        with open(src_path, 'r', encoding='utf-8') as f_src, \
             open(tgt_path, 'r', encoding='utf-8') as f_tgt:
            
            src_lines = f_src.readlines()
            tgt_lines = f_tgt.readlines()

        if len(src_lines) != len(tgt_lines):
            logger.warning(f"Line count mismatch: Source={len(src_lines)}, Target={len(tgt_lines)}")
            # Truncate to minimum length
            min_len = min(len(src_lines), len(tgt_lines))
            src_lines = src_lines[:min_len]
            tgt_lines = tgt_lines[:min_len]

        data = []
        for s, t in zip(src_lines, tgt_lines):
            data.append({
                "source": s.strip(),
                "target": t.strip()
            })
            
        logger.info(f"Loaded {len(data)} pairs from parallel TXT files")
        return data

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalizes text: lowercasing, stripping whitespace, removing emojis/special chars if needed.
        """
        import re
        import unicodedata

        if not isinstance(text, str):
            return ""

        # Lowercase and strip
        text = text.lower().strip()

        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)

        # Remove emojis (simple regex range for common emojis)
        # This is valid for many emoji ranges but not exhaustive
        text = re.sub(r'[^\w\s\.,!?\'"-]', '', text)

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def deduplicate(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Removes duplicate source-target pairs from the dataset.
        """
        unique_pairs = set()
        deduplicated_data = []
        
        for item in data:
            pair = (item['source'], item['target'])
            if pair not in unique_pairs:
                unique_pairs.add(pair)
                deduplicated_data.append(item)
        
        logger.info(f"Deduplication: Removed {len(data) - len(deduplicated_data)} duplicates. Remaining: {len(deduplicated_data)}")
        return deduplicated_data

    def get_tokenizer(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        """
        Returns the NLLB tokenizer.
        """
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            raise
