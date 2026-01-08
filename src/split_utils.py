import random
from typing import List, Dict, Tuple
from src.utils import logger

def split_data(data: List[Dict[str, str]], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[List, List, List]:
    """
    Splits data into train, validation, and test sets.
    """
    if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
        raise ValueError("Ratios must sum to 1.0")

    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    logger.info(f"Data Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

if __name__ == "__main__":
    # Example usage (would typically be called from a main script)
    pass
