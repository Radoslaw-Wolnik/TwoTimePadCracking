import numpy as np


def evaluate_recovery(original1, original2, recovered1, recovered2):
    # Convert to byte arrays
    orig1 = np.frombuffer(original1, dtype=np.uint8)
    orig2 = np.frombuffer(original2, dtype=np.uint8)
    rec1 = np.frombuffer(recovered1, dtype=np.uint8)
    rec2 = np.frombuffer(recovered2, dtype=np.uint8)

    # Byte-level accuracy
    correct_bytes = np.sum((orig1 == rec1) & (orig2 == rec2))
    byte_acc = correct_bytes / len(orig1)

    # Pair-wise accuracy (correct bytes in either position)
    correct_pairs = 0
    for i in range(len(orig1)):
        if (orig1[i] == rec1[i] and orig2[i] == rec2[i]) or \
                (orig1[i] == rec2[i] and orig2[i] == rec1[i]):
            correct_pairs += 1
    pair_acc = correct_pairs / len(orig1)

    return {
        "byte_accuracy": byte_acc,
        "pair_accuracy": pair_acc,
        "switched_positions": np.where(orig1 != rec1)[0]
    }