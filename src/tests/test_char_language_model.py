import math
import pytest
from src.model.char_language_model import CharLanguageModel


def test_train_save_load_roundtrip(tmp_path):
    """Test basic training, saving, and loading"""
    model = CharLanguageModel(n=4)
    text = "ABCDABCD"
    model.train(text)

    path = tmp_path / "m.bin"
    model.save(path)

    # Verify file exists
    assert path.exists()

    # Load and verify basic properties
    loaded = CharLanguageModel.load(path)
    assert loaded.n == 4
    assert len(loaded.vocab) > 0

    # Check log_prob is numeric and finite for a seen char/context
    ctx = b'\x01\x41\x42'  # BOM + bytes representing 'A','B'
    lp = loaded.log_prob(ord('C'), ctx)
    assert math.isfinite(lp)


def test_model_training_learns_patterns():
    """Test that the model actually learns from training data"""
    model = CharLanguageModel(n=4)
    training_text = "hello world hello there"
    model.train(training_text)

    # Test that common sequences have reasonable probabilities
    ctx = b'\x01hel'  # Context after BOM
    prob_lo = model.log_prob(ord('l'), ctx)
    prob_unlikely = model.log_prob(ord('x'), ctx)  # Unlikely character

    # The likely character should have higher probability (less negative log prob)
    assert prob_lo > prob_unlikely


def test_empty_training():
    """Test model behavior with empty training data"""
    model = CharLanguageModel(n=4)
    model.train("")

    # Should handle empty vocab gracefully
    assert len(model.vocab) == 0
    lp = model.log_prob(ord('a'), b'')
    assert math.isfinite(lp)


def test_boundary_markers():
    """Test that BOM/EOM markers are handled correctly"""
    model = CharLanguageModel(n=4)
    text = "tests"
    model.train(text)

    # Should have learned something about BOM context
    assert len(model.vocab) > 0


def test_backoff_behavior():
    """Test that backoff smoothing works for unseen contexts"""
    model = CharLanguageModel(n=4)
    model.train("abcdefgh")

    # Test with completely unseen context
    unseen_context = b'xyz'
    lp = model.log_prob(ord('a'), unseen_context)
    assert math.isfinite(lp)  # Should not crash


def test_save_load_preserves_probabilities(tmp_path):
    """Test that saved and loaded models give same probabilities"""
    original = CharLanguageModel(n=4)
    original.train("This is a tests sentence for model verification.")

    path = tmp_path / "model.bin"
    original.save(path)
    loaded = CharLanguageModel.load(path)

    # Test a few probabilities to ensure they match
    test_cases = [
        (ord('t'), b'\x01Thi'),
        (ord('e'), b'his '),
        (ord('s'), b'is i')
    ]

    for char, context in test_cases:
        orig_prob = original.log_prob(char, context)
        loaded_prob = loaded.log_prob(char, context)

        # Allow small floating point differences
        assert abs(orig_prob - loaded_prob) < 1e-10


def test_different_context_lengths():
    """Test model with different n-gram sizes"""
    for n in [3, 5, 7]:
        model = CharLanguageModel(n=n)
        model.train("Testing different context lengths")
        assert model.n == n

        # Should be able to compute probabilities
        ctx = b'\x01Te'[:n - 1]  # Adjust context length
        lp = model.log_prob(ord('s'), ctx.ljust(n - 1, b' '))
        assert math.isfinite(lp)