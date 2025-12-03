import pytest
import numpy as np  # ADD THIS IMPORT
from pathlib import Path
from src.model.char_language_model import CharLanguageModel


def test_complete_training_pipeline(tmp_path):
    """Test the complete training and saving pipeline"""
    # Create training data
    training_texts = [
        "This is the first training document.",
        "Another document for model training.",
        "Final example text to improve the language model."
    ]

    # Train model
    model = CharLanguageModel(n=5)
    for text in training_texts:
        model.train(text)

    # Verify model learned something
    assert len(model.vocab) > 0
    assert len(model.ngram_counts) > 0

    # Save model
    model_path = tmp_path / "trained_model.bin"
    model.save(model_path)
    assert model_path.exists()

    # Load model
    loaded_model = CharLanguageModel.load(model_path)

    # Verify loaded model works
    assert loaded_model.n == 5
    test_prob = loaded_model.log_prob(ord('e'), b"Th")
    assert isinstance(test_prob, float)
    assert not np.isnan(test_prob)  # Now np is defined


def test_model_incremental_training():
    """Test that model can be trained incrementally"""
    model = CharLanguageModel(n=4)

    # First training
    model.train("First batch of text")
    initial_vocab_size = len(model.vocab)

    # Additional training
    model.train("Second batch with new words")
    final_vocab_size = len(model.vocab)

    # Vocabulary should grow
    assert final_vocab_size >= initial_vocab_size

    # Should still compute probabilities
    prob = model.log_prob(ord('t'), b"Fir")
    assert np.isfinite(prob)