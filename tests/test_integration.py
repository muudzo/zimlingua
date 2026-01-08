import pytest
from unittest.mock import MagicMock, patch
from src.translator import Translator

@pytest.fixture
def mock_translator():
    with patch('src.translator.ctranslate2.Translator') as mock_ct2, \
         patch('src.translator.transformers.AutoTokenizer.from_pretrained') as mock_tok:
        
        # Setup Mock Tokenizer
        tokenizer = MagicMock()
        tokenizer.tokenize.return_value = ["hello", "world"]
        tokenizer.convert_tokens_to_ids.return_value = [1, 2]
        tokenizer.decode.return_value = "mhoro nyika"
        tokenizer.src_lang = "eng_Latn"
        mock_tok.return_value = tokenizer
        
        # Setup Mock CT2 Translator
        translator_instance = MagicMock()
        translation_result = MagicMock()
        translation_result.hypotheses = [["mhoro", "nyika"]]
        translator_instance.translate_batch.return_value = [translation_result]
        mock_ct2.return_value = translator_instance
        
        yield mock_ct2, mock_tok

def test_offline_translation_flow(mock_translator):
    """
    Verifies the translation pipeline (loading, tokenization, inference, detokenization)
    using mocked model/engine to avoid needing heavy weights in CI.
    """
    with patch('src.translator.Path.exists', return_value=True):
        translator = Translator(model_path="dummy_path")
        
        # Test batch translation
        sources = ["Hello World"]
        results = translator.translate_batch(sources, source_lang="eng_Latn", target_lang="sna_Latn")
        
        assert len(results) == 1
        assert results[0] == "mhoro nyika"
        
        # Verify calls
        translator.tokenizer.tokenize.assert_called()
        translator.translator.translate_batch.assert_called()

def test_language_code_mapping():
    # Only tests the helper, no mocks needed technically if config loaded, 
    # but we can rely on default config
    with patch('src.translator.Path.exists', return_value=True), \
         patch('src.translator.ctranslate2.Translator'), \
         patch('src.translator.transformers.AutoTokenizer.from_pretrained'):
        
        t = Translator(model_path="dummy")
        assert t.get_language_code('en') == 'eng_Latn'
        assert t.get_language_code('sn') == 'sna_Latn'
        assert t.get_language_code('xyz') == 'xyz'
