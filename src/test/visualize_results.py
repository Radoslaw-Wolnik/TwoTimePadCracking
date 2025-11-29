# visualize_results.py
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_comparison(byte_accuracies, pair_accuracies):
    """Plot accuracy comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Byte accuracy distribution
    ax1.hist(byte_accuracies, bins=20, alpha=0.7, color='skyblue')
    ax1.axvline(np.mean(byte_accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(byte_accuracies):.2%}')
    ax1.set_xlabel('Byte Accuracy')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Byte Accuracy Distribution')
    ax1.legend()

    # Pair accuracy distribution
    ax2.hist(pair_accuracies, bins=20, alpha=0.7, color='lightgreen')
    ax2.axvline(np.mean(pair_accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(pair_accuracies):.2%}')
    ax2.set_xlabel('Pair Accuracy')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Pair Accuracy Distribution')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('accuracy_results.png')
    plt.show()


def plot_accuracy_over_text_length(test_results):
    """Plot accuracy vs text length"""
    lengths = [r['length'] for r in test_results]
    byte_accs = [r['byte_accuracy'] for r in test_results]
    pair_accs = [r['pair_accuracy'] for r in test_results]

    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, byte_accs, alpha=0.6, label='Byte Accuracy', color='blue')
    plt.scatter(lengths, pair_accs, alpha=0.6, label='Pair Accuracy', color='red')
    plt.xlabel('Text Length (bytes)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Text Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('accuracy_vs_length.png')
    plt.show()