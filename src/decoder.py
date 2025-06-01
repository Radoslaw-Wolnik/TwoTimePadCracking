BOM = b'\x01'  # Start-of-heading character
EOM = b'\x02'   # Start-of-text character

import numpy as np
from .decoder_cy import process_byte_cy  # Cython-accelerated function


# Precomputed probability table structure
class ProbTable:
    def __init__(self, data, size):
        self.data = data  # 1D numpy array of probabilities
        self.size = size  # Size of context space


class TwoTimePadDecoder:
    def __init__(self, model1, model2, beam_width=100):
        self.model1 = model1
        self.model2 = model2
        self.beam_width = beam_width

        # Precompute probability tables
        self.table1 = self._precompute_prob_table(model1)
        self.table2 = self._precompute_prob_table(model2)

        # Context mask (48 bits for 6-byte context)
        self.context_mask = (1 << 48) - 1

    def _precompute_prob_table(self, model):
        """Precompute probability table for efficient access"""
        # In practice, we'd only precompute for observed contexts
        # For simplicity, we'll create a placeholder table
        size = 2 ** 16  # Reduced size for demo (should be 256^6 in production)
        table = np.zeros((size, 256), dtype=np.float32)

        # Populate with dummy probabilities (real implementation would use model)
        table.fill(1e-6)  # Default low probability
        return ProbTable(table, size)

    def decode(self, xor_stream):
        # Initialize with BOM context (integer representation)
        context0 = bytes_to_int(b'\x01' * 6)
        beam = [{
            'context1': context0,
            'context2': context0,
            'log_prob': 0.0,
            'path': []
        }]

        for i, byte in enumerate(xor_stream):
            new_beam = []
            for state in beam:
                # Use Cython for inner loop
                results = process_byte_cy(
                    state,
                    byte,
                    self.table1.data,
                    self.table2.data,
                    self.table1.size,
                    self.context_mask
                )
                new_beam.extend(results)

            # Prune beam
            new_beam.sort(key=lambda x: x['log_prob'], reverse=True)
            beam = new_beam[:self.beam_width]

        # Return best path
        best_state = max(beam, key=lambda x: x['log_prob'])
        p1 = bytes([b[0] for b in best_state['path']])
        p2 = bytes([b[1] for b in best_state['path']])
        return p1, p2


# Helper functions
def bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, 'big')


def int_to_bytes(i: int, length: int) -> bytes:
    return i.to_bytes(length, 'big')
