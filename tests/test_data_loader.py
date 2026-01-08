import pytest
from src.data_loader import DataLoader

@pytest.fixture
def loader():
    return DataLoader()

def test_normalization(loader):
    raw_text = "  Hello  World! 😊 "
    expected = "hello world!"
    assert loader.normalize_text(raw_text) == expected
    
    # Test collapse spaces
    assert loader.normalize_text("a    b") == "a b"

def test_deduplication(loader):
    data = [
        {"source": "hello", "target": "mhoro"},
        {"source": "hello", "target": "mhoro"},
        {"source": "goodbye", "target": "chisarai"}
    ]
    deduplicated = loader.deduplicate(data)
    assert len(deduplicated) == 2
    assert {"source": "hello", "target": "mhoro"} in deduplicated

def test_length_filtering(loader):
    data = [
        {"source": "short", "target": "short"}, # Ratio 1.0 (Keep)
        {"source": "very very very long phrase", "target": "word"}, # High ratio (Drop)
    ]
    filtered = loader.filter_by_length_ratio(data, threshold=2.0)
    assert len(filtered) == 1
    assert filtered[0]['source'] == "short"
