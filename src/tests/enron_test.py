import pathlib

import numpy as np
import pytest
import os
import random
from pathlib import Path

from src.model import CharLanguageModel, TwoTimePadDecoder
from src.model.evaluate import evaluate_recovery
from src.tests.conftest import MIN_EMAILS_FOR_TEST

# Directory of THIS test file
TEST_DIR = Path(__file__).parent

EMAILS_DIR = (TEST_DIR / "data" / "10k_processed_emails").resolve()

def load_enron_emails(data_dir: pathlib.Path, max_emails: int = 1000):
    """Load Enron emails from directory"""
    emails = []

    for file_path in data_dir.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if len(content) > 100:  # Only use emails with substantial content
                    emails.append(content.encode('utf-8'))
        except Exception as e:
            continue

        if len(emails) >= max_emails:
            break

    return emails


@pytest.mark.veryslow
def test_decoder_real_emails_same_domain():
    """Test with real Enron emails from same domain"""
    # pytest.skip("Skip slow test - 6min for this one")
    emails = load_enron_emails(EMAILS_DIR)

    if len(emails) < MIN_EMAILS_FOR_TEST['test_decoder_real_emails_same_domain']:
        pytest.skip(f"Need at least {MIN_EMAILS_FOR_TEST['test_decoder_real_emails_same_domain']} emails")

    # Use first 80% for training, last 20% for testing
    split_idx = int(0.8 * len(emails))
    train_emails = emails[:split_idx]
    test_emails = emails[split_idx:]

    # Train model on training emails
    model = CharLanguageModel(n=6)
    for email in train_emails:
        model.train(email)

    # Test on pairs from tests set
    successful_decodes = 0
    total_tests = min(10, len(test_emails) // 2)

    for i in range(total_tests):
        text1 = test_emails[i * 2]
        text2 = test_emails[i * 2 + 1]

        # Ensure same length
        min_len = min(len(text1), len(text2), 200)  # Cap at 200 bytes for performance
        text1 = text1[:min_len]
        text2 = text2[:min_len]

        xor_stream = [a ^ b for a, b in zip(text1, text2)]
        decoder = TwoTimePadDecoder(model, model, beam_width=200)
        recovered1, recovered2 = decoder.decode(xor_stream)

        results = evaluate_recovery(text1, text2, recovered1, recovered2)

        print(f"Test {i + 1}: Byte Acc: {results['byte_accuracy']:.2%}, "
              f"Pair Acc: {results['pair_accuracy']:.2%}")

        if results['pair_accuracy'] > 0.7:  # Reasonable threshold for real data
            successful_decodes += 1

    success_rate = successful_decodes / total_tests
    print(f"Success rate: {success_rate:.2%} ({successful_decodes}/{total_tests})")
    assert success_rate > 0.5  # Should succeed on majority of tests

@pytest.mark.veryslow
def test_decoder_different_domains():
    """Test with models trained on different email subsets"""
    emails = load_enron_emails(EMAILS_DIR, max_emails=2002)

    if len(emails) < MIN_EMAILS_FOR_TEST['test_decoder_different_domains']:
        pytest.skip(f"Need at least {MIN_EMAILS_FOR_TEST['test_decoder_different_domains']} emails")

    # Split emails into two different domains (by sender or random split)
    random.shuffle(emails)

    # Use 80% for training (split between domains)
    split_idx = int(0.8 * len(emails))
    train_emails = emails[:split_idx]
    test_emails = emails[split_idx:]  # 20% for testing - NEVER seen during training

    # Split training emails into two domains
    train_split = len(train_emails) // 2
    domain_a_train = train_emails[:train_split]
    domain_b_train = train_emails[train_split:]

    # Split testing emails too
    test_split = len(test_emails) // 2
    domain_a_test = test_emails[:test_split]
    domain_b_test = test_emails[test_split:]

    # Check we have enough in each domain
    if len(domain_a_train) < 800 or len(domain_b_train) < 800:
        pytest.skip(f"Not enough emails in each domain: {len(domain_a_train)} and {len(domain_b_train)}")

    # Train separate models
    model_a = CharLanguageModel(n=7)
    for email in domain_a_train[:800]:  # Use subset for training
        model_a.train(email)

    model_b = CharLanguageModel(n=7)
    for email in domain_b_train[:800]:
        model_b.train(email)

    min_test_size = 50  # Minimum test emails needed

    if len(domain_a_test) < min_test_size or len(domain_b_test) < min_test_size:
        pytest.skip(f"Need at least {min_test_size} test emails in each domain")

    # Run multiple tests
    num_tests = 20
    pair_accuracies = []
    byte_accuracies = []

    for i in range(num_tests):
        # Skip if not enough test emails
        if not domain_a_test or not domain_b_test:
            continue

        text1 = random.choice(domain_a_test)
        text2 = random.choice(domain_b_test)

        min_len = min(len(text1), len(text2), 150)
        if min_len < 50:  # Skip very short texts
            continue

        text1 = text1[:min_len]
        text2 = text2[:min_len]

        xor_stream = [a ^ b for a, b in zip(text1, text2)]
        decoder = TwoTimePadDecoder(model_a, model_b, beam_width=500, n=7)
        recovered1, recovered2 = decoder.decode(xor_stream)

        results = evaluate_recovery(text1, text2, recovered1, recovered2)

        pair_accuracies.append(results['pair_accuracy'])
        byte_accuracies.append(results['byte_accuracy'])

    if not pair_accuracies:
        pytest.skip("No valid test cases could be run")

    # 6. Calculate averages
    avg_pair = sum(pair_accuracies) / len(pair_accuracies)
    avg_byte = sum(byte_accuracies) / len(byte_accuracies)

    # 7. Print summary
    print(f"\nCross-domain Decoding Results (average of {len(pair_accuracies)} tests):")
    print(f"Average Pair Accuracy: {avg_pair:.2%}")
    print(f"Average Byte Accuracy: {avg_byte:.2%}")
    print(f"Pair Accuracy Range: {min(pair_accuracies):.2%} - {max(pair_accuracies):.2%}")

    # 8. Assertions
    # Cross-domain is harder, use appropriate thresholds
    assert avg_pair > 0.5, f"Average pair accuracy {avg_pair:.2%} too low"
    # Also check that at least some tests had good results
    assert max(pair_accuracies) > 0.7, f"No test achieved >70% pair accuracy"


@pytest.mark.veryslow
def test_decoder_various_lengths():
    """Test decoding with emails of various lengths"""
    emails = load_enron_emails(EMAILS_DIR)

    if len(emails) < MIN_EMAILS_FOR_TEST['test_decoder_various_lengths']:
        pytest.skip(f"Need at least {MIN_EMAILS_FOR_TEST['test_decoder_various_lengths']} emails")

    model = CharLanguageModel(n=6)
    for email in emails[:30]:  # Train on first 30 emails
        model.train(email)

    test_cases = [
        (50, 50),  # Very short
        (100, 100),  # Short
        (200, 200),  # Medium
        (300, 300),  # Longer
    ]

    for target_len1, target_len2 in test_cases:
        # Find emails close to target lengths
        text1 = next((e for e in emails[30:] if len(e) >= target_len1), None)
        text2 = next((e for e in emails[30:] if len(e) >= target_len2 and e != text1), None)

        if not text1 or not text2:
            continue

        text1 = text1[:target_len1]
        text2 = text2[:target_len2]

        min_len = min(len(text1), len(text2))
        text1 = text1[:min_len]
        text2 = text2[:min_len]

        xor_stream = [a ^ b for a, b in zip(text1, text2)]
        decoder = TwoTimePadDecoder(model, model, beam_width=200)
        recovered1, recovered2 = decoder.decode(xor_stream)

        results = evaluate_recovery(text1, text2, recovered1, recovered2)

        print(f"Length {min_len} - Byte Acc: {results['byte_accuracy']:.2%}, "
              f"Pair Acc: {results['pair_accuracy']:.2%}")

        # Longer texts should generally have lower accuracy
        if min_len <= 50:
            assert results['pair_accuracy'] > 0.4 # we have 44% acc
        elif min_len <= 100:
            assert results['pair_accuracy'] > 0.35 # we have 39% acc
        else:
            assert results['pair_accuracy'] >= 0.25 # we have 25%


def test_evaluate_recovery_accuracy():
    """Test that the evaluation function is working correctly"""
    # Perfect recovery
    text1 = b"Hello World"
    text2 = b"Test Message"
    recovered1 = b"Hello World"
    recovered2 = b"Test Message"

    results = evaluate_recovery(text1, text2, recovered1, recovered2)
    assert results['byte_accuracy'] == 1.0
    assert results['pair_accuracy'] == 1.0

    # Partial recovery - make texts same length for clear comparison
    text1 = b"Hello World"  # 11 bytes
    text2 = b"TestMessage"  # 11 bytes (removed space to make same length)
    recovered1 = b"Hello Xorld"  # 1 byte wrong (10 correct)
    recovered2 = b"TestMessage"  # perfect (11 correct)

    results = evaluate_recovery(text1, text2, recovered1, recovered2)
    byte_acc = 21 / 22  # 21 correct out of 22 total bytes (11 + 11)
    pair_acc = 10 / 11  # 10 correct pairs out of 11 positions
    assert abs(results['byte_accuracy'] - byte_acc) < 0.001
    assert abs(results['pair_accuracy'] - pair_acc) < 0.001