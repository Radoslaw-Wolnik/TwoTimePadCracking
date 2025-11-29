# test_model.py
import os
import random
from src import CharLanguageModel, TwoTimePadDecoder, evaluate_recovery


def create_test_pair(corpus_dir, output_dir):
    """Create a test pair of emails and their XOR"""
    # Get all email files
    email_files = [f for f in os.listdir(corpus_dir) if f.endswith('.txt')]

    # Pick two random emails
    email1_file, email2_file = random.sample(email_files, 2)

    # Read the emails
    with open(os.path.join(corpus_dir, email1_file), 'rb') as f:
        email1 = f.read()
    with open(os.path.join(corpus_dir, email2_file), 'rb') as f:
        email2 = f.read()

    # Truncate to same length (take the shorter one)
    min_len = min(len(email1), len(email2))
    email1 = email1[:min_len]
    email2 = email2[:min_len]

    # Create XOR stream
    xor_stream = bytes(a ^ b for a, b in zip(email1, email2))

    # Save test data
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "original1.bin"), 'wb') as f:
        f.write(email1)
    with open(os.path.join(output_dir, "original2.bin"), 'wb') as f:
        f.write(email2)
    with open(os.path.join(output_dir, "xor_stream.bin"), 'wb') as f:
        f.write(xor_stream)

    return email1, email2, xor_stream


def test_model_single_pair(model_path, test_dir):
    """Test the model on a single pair"""
    # Load model
    model = CharLanguageModel.load(model_path)

    # Load test data
    with open(os.path.join(test_dir, "original1.bin"), 'rb') as f:
        original1 = f.read()
    with open(os.path.join(test_dir, "original2.bin"), 'rb') as f:
        original2 = f.read()
    with open(os.path.join(test_dir, "xor_stream.bin"), 'rb') as f:
        xor_stream = f.read()

    # Decode
    decoder = TwoTimePadDecoder(model, model, beam_width=100)
    recovered1, recovered2 = decoder.decode(xor_stream)

    # Evaluate
    results = evaluate_recovery(original1, original2, recovered1, recovered2)

    print("=== Single Test Results ===")
    print(f"Byte Accuracy: {results['byte_accuracy']:.2%}")
    print(f"Pair Accuracy: {results['pair_accuracy']:.2%}")
    print(f"Length: {len(original1)} bytes")

    # Show sample of recovered text
    print("\n=== Sample of Original Email 1 ===")
    print(original1[:200].decode('utf-8', errors='replace'))
    print("\n=== Sample of Recovered Email 1 ===")
    print(recovered1[:200].decode('utf-8', errors='replace'))

    return results


def test_model_multiple_pairs(model_path, corpus_dir, num_tests=10):
    """Test the model on multiple random pairs"""
    model = CharLanguageModel.load(model_path)
    decoder = TwoTimePadDecoder(model, model, beam_width=100)

    byte_accuracies = []
    pair_accuracies = []

    for i in range(num_tests):
        print(f"Running test {i + 1}/{num_tests}...")

        # Create test pair
        email1, email2, xor_stream = create_test_pair(corpus_dir, f"test_temp_{i}")

        # Decode
        recovered1, recovered2 = decoder.decode(xor_stream)

        # Evaluate
        results = evaluate_recovery(email1, email2, recovered1, recovered2)

        byte_accuracies.append(results['byte_accuracy'])
        pair_accuracies.append(results['pair_accuracy'])

        print(f"  Test {i + 1}: Byte Acc: {results['byte_accuracy']:.2%}, Pair Acc: {results['pair_accuracy']:.2%}")

        # Clean up
        os.remove(f"test_temp_{i}/original1.bin")
        os.remove(f"test_temp_{i}/original2.bin")
        os.remove(f"test_temp_{i}/xor_stream.bin")
        os.rmdir(f"test_temp_{i}")

    print("\n=== Overall Results ===")
    print(f"Average Byte Accuracy: {sum(byte_accuracies) / len(byte_accuracies):.2%}")
    print(f"Average Pair Accuracy: {sum(pair_accuracies) / len(pair_accuracies):.2%}")
    print(f"Best Byte Accuracy: {max(byte_accuracies):.2%}")
    print(f"Worst Byte Accuracy: {min(byte_accuracies):.2%}")

    return byte_accuracies, pair_accuracies


if __name__ == "__main__":
    model_path = "email_model.bin"  # Change to your model path
    corpus_dir = "processed_emails"  # Change to your email corpus

    # Test single pair
    print("Testing single pair...")
    test_model_single_pair(model_path, "test_data")

    # Test multiple pairs
    print("\n" + "=" * 50)
    print("Testing multiple pairs...")
    test_model_multiple_pairs(model_path, corpus_dir, num_tests=5)