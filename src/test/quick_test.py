# quick_test.py
from src import CharLanguageModel


def quick_model_validation(model_path):
    """Quick test to see if the model loads and works"""
    try:
        # Load model
        model = CharLanguageModel.load(model_path)
        print("✓ Model loaded successfully")

        # Test basic probability calculation
        test_context = b'\x01' * 6  # BOM context
        test_char = ord('H')  # 'H' character

        prob = model.log_prob(test_char, test_context)
        print(f"✓ Probability calculation works: P('H'|context) = {prob}")

        # Test with some real text context
        test_context2 = b'Hello '  # 6 bytes
        prob2 = model.log_prob(ord('W'), test_context2)
        print(f"✓ Contextual probability works: P('W'|'Hello ') = {prob2}")

        # Check model stats
        print(f"✓ Model n-gram size: {model.n}")
        print(f"✓ Vocabulary size: {len(model.vocab)}")
        print(f"✓ Contexts in model: {len(model.context_counts)}")

        return True

    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False


if __name__ == "__main__":
    quick_model_validation("email_model.bin")