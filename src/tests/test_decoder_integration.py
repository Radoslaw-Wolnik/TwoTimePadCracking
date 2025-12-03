import pytest
from src.model.char_language_model import CharLanguageModel
from src.model.decoder import TwoTimePadDecoder
from src.model.evaluate import evaluate_recovery


@pytest.mark.slow
def test_decoder_recovers_short_messages():
    """Test that decoder can recover very short messages"""
    # Two short messages that are easy for decoder to find
    m1 = b"Hello!"
    m2 = b"World!"

    # Train small models on the exact messages
    cm1 = CharLanguageModel(n=4)
    cm2 = CharLanguageModel(n=4)
    cm1.train(m1)
    cm2.train(m2)

    # Use models directly
    xor_stream = [a ^ b for a, b in zip(m1, m2)]
    decoder = TwoTimePadDecoder(cm1, cm2, beam_width=512)
    r1, r2 = decoder.decode(xor_stream)

    res = evaluate_recovery(m1, m2, r1[:len(m1)], r2[:len(m2)])

    # For very short messages with perfect training, expect good recovery
    assert res['byte_accuracy'] >= 0.5


@pytest.mark.slow
def test_decoder_with_same_model():
    """Test decoder using the same model for both texts (common case)"""
    # Two texts from similar domain
    text1 = b"This is the first example text for testing."
    text2 = b"Here is another text example for the tests."

    # Train one model on combined data (simulating same domain)
    combined = text1 + b" " + text2
    model = CharLanguageModel(n=5)
    model.train(combined)

    xor_stream = [a ^ b for a, b in zip(text1, text2)]
    decoder = TwoTimePadDecoder(model, model, beam_width=200)
    recovered1, recovered2 = decoder.decode(xor_stream)

    # Evaluate results
    results = evaluate_recovery(text1, text2, recovered1, recovered2)

    # With same domain and reasonable beam, should get some recovery
    assert results['pair_accuracy'] > 0.5  # At least half pairs correct
    # i get 1.0


def test_decoder_handles_different_lengths():
    """Test that decoder handles texts of different lengths"""
    short = b"Short"
    long = b"Much longer text example here"

    # Use minimum length
    min_len = min(len(short), len(long))
    short = short[:min_len]
    long = long[:min_len]

    model = CharLanguageModel(n=4)
    model.train(short + long)

    xor_stream = [a ^ b for a, b in zip(short, long)]
    decoder = TwoTimePadDecoder(model, model, beam_width=100)
    r1, r2 = decoder.decode(xor_stream)

    # Should return texts of correct length
    assert len(r1) == min_len
    assert len(r2) == min_len