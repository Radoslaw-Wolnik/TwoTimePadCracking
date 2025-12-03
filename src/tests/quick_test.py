#!/usr/bin/env python3
"""
Pytest-compatible quick manual tests of the decoding pipeline
"""

import pytest
from src.model.char_language_model import CharLanguageModel
from src.model.decoder import TwoTimePadDecoder
from src.model.evaluate import evaluate_recovery


def test_quick_decoding_pipeline():
    """Quick manual tests of the complete decoding pipeline"""
    # Test data - use similar texts for better results
    text1 = b"This is a tests message for verification."
    text2 = b"This is another tests text for checking."

    print("Original texts:")
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")

    # Train models on each text separately
    model1 = CharLanguageModel(n=5)
    model2 = CharLanguageModel(n=5)
    model1.train(text1)
    model2.train(text2)

    print(f"Model 1 vocab: {len(model1.vocab)}, Model 2 vocab: {len(model2.vocab)}")

    # Create XOR stream
    min_len = min(len(text1), len(text2))
    xor_stream = [text1[i] ^ text2[i] for i in range(min_len)]
    print(f"XOR stream length: {len(xor_stream)}")

    # Try to decode
    decoder = TwoTimePadDecoder(model1, model2, beam_width=200)
    print("\nDecoding...")
    recovered1, recovered2 = decoder.decode(xor_stream)

    # Evaluate only the overlapping part
    results = evaluate_recovery(
        text1[:min_len],
        text2[:min_len],
        recovered1,
        recovered2
    )

    print("\nResults:")
    print(f"Byte Accuracy: {results['byte_accuracy']:.2%}")
    print(f"Pair Accuracy: {results['pair_accuracy']:.2%}")
    print(f"Switches: {results['total_switches']}")

    print(f"\nRecovered Text 1: {recovered1}")
    print(f"Recovered Text 2: {recovered2}")

    # Assert correct lengths
    assert len(recovered1) == min_len, f"Expected {min_len}, got {len(recovered1)}"
    assert len(recovered2) == min_len, f"Expected {min_len}, got {len(recovered2)}"


def test_decoding_with_different_beam_widths():
    """Test how beam width affects decoding accuracy"""
    # Use more similar texts for better results
    text1 = b"Short tests message one for testing."
    text2 = b"Short tests message two for checking."

    # Train model on combined data
    model = CharLanguageModel(n=5)
    model.train(text1 + b" " + text2)

    # Use min length for XOR
    min_len = min(len(text1), len(text2))
    xor_stream = [text1[i] ^ text2[i] for i in range(min_len)]

    # Test different beam widths
    beam_widths = [50, 100, 200]  # Start with reasonable sizes
    accuracies = []

    for beam_width in beam_widths:
        decoder = TwoTimePadDecoder(model, model, beam_width=beam_width)
        recovered1, recovered2 = decoder.decode(xor_stream)

        # Only evaluate the overlapping part
        results = evaluate_recovery(
            text1[:min_len],
            text2[:min_len],
            recovered1,
            recovered2
        )
        accuracies.append(results['pair_accuracy'])

        # Basic assertions - lengths should match XOR stream length
        assert len(recovered1) == min_len, f"Expected {min_len}, got {len(recovered1)}"
        assert len(recovered2) == min_len, f"Expected {min_len}, got {len(recovered2)}"

    print(f"Accuracies with different beam widths: {list(zip(beam_widths, accuracies))}")

    # For similar texts, we should get reasonable accuracy with sufficient beam
    assert max(accuracies) > 0.1, f"Expected at least 10% accuracy, got {max(accuracies)}"


@pytest.mark.slow
def test_decoding_with_larger_texts():
    """Test decoding with slightly larger texts"""
    text1 = b"This is a longer tests message that contains more text and should provide a better tests of the decoding capabilities of our system."
    text2 = b"Another longer example text that differs significantly from the first one but still maintains reasonable English language structure and common words."

    model = CharLanguageModel(n=6)
    model.train(text1)
    model.train(text2)

    # ensure equal length by padding with space
    if len(text1) > len(text2):
        text2 = text2.ljust(len(text1), b' ')
    elif len(text2) > len(text1):
        text1 = text1.ljust(len(text2), b' ')

    xor_stream = [a ^ b for a, b in zip(text1, text2)]
    decoder = TwoTimePadDecoder(model, model, beam_width=150)
    recovered1, recovered2 = decoder.decode(xor_stream)

    results = evaluate_recovery(text1, text2, recovered1, recovered2)

    print(f"Large text results - Byte Acc: {results['byte_accuracy']:.2%}, "
          f"Pair Acc: {results['pair_accuracy']:.2%}")

    assert len(recovered1) == len(text1)
    assert len(recovered2) == len(text2)
    assert results['pair_accuracy'] > 0  # Should get at least some pairs correct


def test_model_serialization_roundtrip(tmp_path):
    """Test that models can be saved and loaded properly"""
    # Original training data
    text1 = b"Test message for serialization."
    text2 = b"Another text for model testing."

    # Create and train original model
    original_model = CharLanguageModel(n=5)
    original_model.train(text1)
    original_model.train(text2)

    # Save model
    model_path = tmp_path / "test_model.bin"
    original_model.save(model_path)

    # Load model
    loaded_model = CharLanguageModel.load(model_path)

    # Test that probabilities are consistent
    test_cases = [
        (ord('T'), b'\x01Test'),
        (ord('e'), b'Test'),
        (ord('s'), b'est ')
    ]

    for char, context in test_cases:
        original_prob = original_model.log_prob(char, context)
        loaded_prob = loaded_model.log_prob(char, context)

        # Should be very close (allowing for floating point differences)
        assert abs(original_prob - loaded_prob) < 1e-10

    # Test that loaded model can be used for decoding
    xor_stream = [a ^ b for a, b in zip(text1, text2)]
    decoder = TwoTimePadDecoder(loaded_model, loaded_model, beam_width=100)
    recovered1, recovered2 = decoder.decode(xor_stream)

    # Basic sanity checks
    assert len(recovered1) == len(text1)
    assert len(recovered2) == len(text2)


if __name__ == "__main__":
    # This allows the file to still be run directly for quick testing
    test_quick_decoding_pipeline()
    print("\n✓ All quick tests passed!")