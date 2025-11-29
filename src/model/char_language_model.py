import tempfile

import numpy as np
from collections import defaultdict
import math
import mmap
import os
import struct
import re
import email
from email import policy

# Special control characters
BOM = b'\x01'  # Start-of-heading character
EOM = b'\x02'  # Start-of-text character


class CharLanguageModel:
    def __init__(self, n=7):
        self.n = n
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.backoff_weights = {}
        self.min_prob = 1e-10  # Avoid zero probabilities

    def train(self, text):
        """Train on text with BOM prepended and EOM appended"""
        if not isinstance(text, bytes):
            text = text.encode('utf-8', 'ignore')

        # Add boundary markers
        text = BOM + text + EOM

        # Slide n-gram window through text
        for i in range(len(text) - self.n):
            context = text[i:i + self.n - 1]
            next_char = text[i + self.n - 1]
            self.ngram_counts[context][next_char] += 1
            self.context_counts[context] += 1
            self.vocab.add(next_char)

        # Compute smoothing weights
        self._compute_witten_bell_weights()

    def _compute_witten_bell_weights(self):
        """Calculate backoff weights for all contexts"""
        for context, total_count in self.context_counts.items():
            distinct_chars = len(self.ngram_counts[context])
            # λ = distinct_chars / (distinct_chars + total_count)
            self.backoff_weights[context] = distinct_chars / (distinct_chars + total_count)

    def log_prob(self, char, context):
        """Recursive probability with Witten-Bell smoothing"""
        # Base case: uniform distribution for empty context
        if len(context) == 0:
            return math.log(1 / len(self.vocab))

        # Full context exists in training data
        if context in self.ngram_counts:
            char_count = self.ngram_counts[context].get(char, 0)
            if char_count > 0:
                return math.log(char_count / self.context_counts[context])

        # Backoff to shorter context
        backoff_weight = self.backoff_weights.get(context, 0.4)
        shorter_context = context[1:]
        backoff_log_prob = self.log_prob(char, shorter_context)
        backoff_prob = math.exp(backoff_log_prob)

        # Apply smoothing with minimum probability floor
        smoothed_prob = (backoff_weight * backoff_prob +
                         (1 - backoff_weight) * self.min_prob)
        return math.log(max(smoothed_prob, self.min_prob))

    def save(self, file_path):
        """Save model to file (atomic write, write header + number of contexts)."""
        # Prepare data to write into a temp file
        fd, temp_path = tempfile.mkstemp(prefix="clm_", suffix=".bin", dir=os.path.dirname(file_path) or ".")
        try:
            with os.fdopen(fd, "wb") as f:
                # Magic (4 bytes) + version (I)
                f.write(b'CLM1')  # magic
                f.write(struct.pack('I', 1))  # version 1

                # n and vocab size
                f.write(struct.pack('II', self.n, len(self.vocab)))

                # number of contexts so loader knows how many entries to read
                f.write(struct.pack('I', len(self.ngram_counts)))

                # Write contexts
                for context, char_counts in self.ngram_counts.items():
                    if len(context) != self.n - 1:
                        # Ensure context length consistent
                        raise ValueError("Context length mismatch when saving.")
                    f.write(context)  # fixed-length bytes
                    f.write(struct.pack('I', self.context_counts[context]))
                    f.write(struct.pack('I', len(char_counts)))
                    for char, count in char_counts.items():
                        f.write(struct.pack('BI', char, count))
            # atomic replace
            os.replace(temp_path, file_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    @classmethod
    def load(cls, file_path):
        """Load model safely and raise informative errors if file malformed."""
        model = cls()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        with open(file_path, 'rb') as f:
            # Read and validate magic/version
            magic = f.read(4)
            if len(magic) < 4:
                raise IOError("Model file too short (missing magic/header).")
            if magic != b'CLM1':
                raise IOError("Unknown model file format or corrupted file (bad magic).")

            ver_bytes = f.read(4)
            if len(ver_bytes) < 4:
                raise IOError("Model file missing version field.")
            version = struct.unpack('I', ver_bytes)[0]
            if version != 1:
                raise IOError(f"Unsupported model version: {version}")

            # Read n, vocab_size
            header = f.read(8)
            if len(header) < 8:
                raise IOError("Model file missing n/vocab_size header.")
            model.n, vocab_size = struct.unpack('II', header)

            # Read number of contexts
            num_ctx_bytes = f.read(4)
            if len(num_ctx_bytes) < 4:
                raise IOError("Model file missing number-of-contexts field.")
            num_contexts = struct.unpack('I', num_ctx_bytes)[0]

            # Read contexts exactly num_contexts times
            for i in range(num_contexts):
                context = f.read(model.n - 1)
                if len(context) < (model.n - 1):
                    raise IOError(f"Unexpected EOF while reading context #{i}")
                total_count_bytes = f.read(4)
                if len(total_count_bytes) < 4:
                    raise IOError(f"Unexpected EOF while reading total_count for context #{i}")
                total_count = struct.unpack('I', total_count_bytes)[0]
                model.context_counts[context] = total_count

                num_chars_bytes = f.read(4)
                if len(num_chars_bytes) < 4:
                    raise IOError(f"Unexpected EOF while reading num_chars for context #{i}")
                num_chars = struct.unpack('I', num_chars_bytes)[0]

                char_counts = {}
                for j in range(num_chars):
                    entry = f.read(5)  # 1-byte char + 4-byte count
                    if len(entry) < 5:
                        raise IOError(f"Unexpected EOF while reading char entry {j} for context #{i}")
                    char, count = struct.unpack('BI', entry)
                    char_counts[char] = count
                    model.vocab.add(char)

                model.ngram_counts[context] = char_counts

        model._compute_witten_bell_weights()
        return model