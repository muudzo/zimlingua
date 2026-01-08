import pytest
from src.config import config

def test_dummy():
    """
    A simple dummy test to verify the test harness is working.
    """
    assert True

def test_config_loader():
    """
    Test that the Singleton config loader works.
    """
    # Assuming the config.yaml exists and has 'model.name'
    model_name = config.get("model.name")
    assert model_name is not None
    assert isinstance(model_name, str)
