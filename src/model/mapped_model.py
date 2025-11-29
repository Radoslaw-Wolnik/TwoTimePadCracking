import numpy as np
import math
import mmap
import os
import struct



class MappedLanguageModel:
    """Memory-mapped efficient probability access"""

    def __init__(self, model_path):
        self.file = open(model_path, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), 0)
        self.header = struct.unpack('II', self.mmap[:8])
        self.n, self.vocab_size = self.header
        self.offset = 8

        # Build context index
        self.context_map = {}
        pos = self.offset
        while pos < len(self.mmap):
            context = self.mmap[pos:pos + self.n - 1]
            pos += self.n - 1
            count = struct.unpack('I', self.mmap[pos:pos + 4])[0]
            pos += 4
            self.context_map[context] = pos
            pos += 256 * 8  # Skip probability table

    def log_prob(self, char, context):
        """Get log probability from memory-mapped file"""
        # Get context position
        context_key = context if isinstance(context, bytes) else bytes(context)
        if context_key not in self.context_map:
            return self._backoff_prob(char, context)

        # Read precomputed probability
        pos = self.context_map[context_key] + char * 8
        return struct.unpack('d', self.mmap[pos:pos + 8])[0]

    def _backoff_prob(self, char, context):
        """Recursive backoff for unseen contexts"""
        if len(context) == 0:
            return math.log(1 / self.vocab_size)

        # Try shorter context
        return self.log_prob(char, context[1:])

    @classmethod
    def build(cls, char_model, output_path):
        """Create memory-mapped version from CharLanguageModel"""
        with open(output_path, 'wb') as f:
            # Write header
            f.write(struct.pack('II', char_model.n, len(char_model.vocab)))

            # Write each context with probability table
            for context in char_model.context_counts:
                # Write context bytes
                f.write(context)

                # Write placeholder for total count (not used in runtime)
                f.write(struct.pack('I', char_model.context_counts[context]))

                # Precompute and write probability table
                probs = np.zeros(256, dtype=np.float64)
                for char in range(256):
                    probs[char] = char_model.log_prob(char, context)
                f.write(probs.tobytes())