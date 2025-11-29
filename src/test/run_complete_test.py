# run_complete_test.py
from src.test.test_model import test_model_multiple_pairs
from src.test.quick_test import quick_model_validation
from src.test.visualize_results import plot_accuracy_comparison


def run_complete_test_suite():
    """Run complete test suite"""
    model_path = "email_model.bin"
    corpus_dir = "processed_emails"

    print("=== Starting Complete Test Suite ===")

    # 1. Quick model validation
    print("\n1. Model Validation")
    if not quick_model_validation(model_path):
        print("Model validation failed. Stopping tests.")
        return

    # 2. Run multiple tests
    print("\n2. Running Accuracy Tests")
    byte_accuracies, pair_accuracies = test_model_multiple_pairs(
        model_path, corpus_dir, num_tests=10
    )

    # 3. Visualize results
    print("\n3. Generating Visualizations")
    plot_accuracy_comparison(byte_accuracies, pair_accuracies)

    # 4. Summary
    print("\n=== Test Summary ===")
    print(f"Model: {model_path}")
    print(f"Tests run: {len(byte_accuracies)}")
    print(f"Average Byte Accuracy: {np.mean(byte_accuracies):.2%} ± {np.std(byte_accuracies):.2%}")
    print(f"Average Pair Accuracy: {np.mean(pair_accuracies):.2%} ± {np.std(pair_accuracies):.2%}")

    # Compare with paper results
    print("\n=== Comparison with Paper ===")
    print("Paper results (email vs email): ~82% byte accuracy")
    print(f"Our results: {np.mean(byte_accuracies):.2%} byte accuracy")

    if np.mean(byte_accuracies) > 0.5:
        print("✓ Model is working reasonably well!")
    else:
        print("⚠ Model needs improvement")


if __name__ == "__main__":
    run_complete_test_suite()