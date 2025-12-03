import numpy as np
from collections import Counter


def evaluate_recovery(original1, original2, recovered1, recovered2):
    """Enhanced evaluation with more metrics"""
    # Handle empty inputs
    if len(original1) == 0 and len(original2) == 0:
        return {
            "byte_accuracy": 0.0,
            "pair_accuracy": 0.0,
            "printable_accuracy_1": 0.0,
            "printable_accuracy_2": 0.0,
            "word_accuracy_1": 0.0,
            "word_accuracy_2": 0.0,
            "switched_positions": [],
            "total_switches": 0,
            "switch_rate": 0.0
        }

    # Convert to byte arrays, using minimum length to avoid shape mismatches
    min_len = min(len(original1), len(original2), len(recovered1), len(recovered2))

    orig1 = np.frombuffer(original1[:min_len], dtype=np.uint8)
    orig2 = np.frombuffer(original2[:min_len], dtype=np.uint8)
    rec1 = np.frombuffer(recovered1[:min_len], dtype=np.uint8)
    rec2 = np.frombuffer(recovered2[:min_len], dtype=np.uint8)

    # FIXED: Byte accuracy - count ALL correct bytes independently
    correct_bytes_1 = np.sum(orig1 == rec1)
    correct_bytes_2 = np.sum(orig2 == rec2)
    total_bytes = len(orig1) + len(orig2)
    byte_accuracy = (correct_bytes_1 + correct_bytes_2) / total_bytes if total_bytes > 0 else 0.0

    # Pair-wise accuracy (both correct OR switched)
    correct_pairs = 0
    switched_positions = []
    for i in range(min_len):
        if (orig1[i] == rec1[i] and orig2[i] == rec2[i]):
            correct_pairs += 1
        elif (orig1[i] == rec2[i] and orig2[i] == rec1[i]):
            correct_pairs += 1
            switched_positions.append(i)

    pair_accuracy = correct_pairs / min_len if min_len > 0 else 0.0

    # Character-level accuracy (printable ASCII)
    printable_chars1 = np.sum((orig1 >= 32) & (orig1 <= 126))
    printable_correct1 = np.sum((orig1 == rec1) & (orig1 >= 32) & (orig1 <= 126))
    printable_acc1 = printable_correct1 / printable_chars1 if printable_chars1 > 0 else 0

    printable_chars2 = np.sum((orig2 >= 32) & (orig2 <= 126))
    printable_correct2 = np.sum((orig2 == rec2) & (orig2 >= 32) & (orig2 <= 126))
    printable_acc2 = printable_correct2 / printable_chars2 if printable_chars2 > 0 else 0

    # Word-level accuracy (approximate)
    def bytes_to_words(byte_arr):
        text = byte_arr.tobytes().decode('utf-8', errors='ignore')
        return text.split()

    words1_orig = bytes_to_words(orig1)
    words1_rec = bytes_to_words(rec1)
    words2_orig = bytes_to_words(orig2)
    words2_rec = bytes_to_words(rec2)

    # Simple word accuracy (first min(len) words)
    min_words1 = min(len(words1_orig), len(words1_rec))
    word_correct1 = sum(1 for i in range(min_words1) if words1_orig[i] == words1_rec[i])
    word_acc1 = word_correct1 / min_words1 if min_words1 > 0 else 0

    min_words2 = min(len(words2_orig), len(words2_rec))
    word_correct2 = sum(1 for i in range(min_words2) if words2_orig[i] == words2_rec[i])
    word_acc2 = word_correct2 / min_words2 if min_words2 > 0 else 0

    return {
        "byte_accuracy": byte_accuracy,
        "pair_accuracy": pair_accuracy,
        "printable_accuracy_1": printable_acc1,
        "printable_accuracy_2": printable_acc2,
        "word_accuracy_1": word_acc1,
        "word_accuracy_2": word_acc2,
        "switched_positions": switched_positions,
        "total_switches": len(switched_positions),
        "switch_rate": len(switched_positions) / min_len if min_len > 0 else 0
    }