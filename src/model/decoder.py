from src.model.char_language_model import BOM


class TwoTimePadDecoder:
    def __init__(self, model1, model2, beam_width=100):
        self.model1 = model1
        self.model2 = model2
        self.beam_width = beam_width
        self.context_mask = (1 << 48) - 1  # 6-byte mask

    def decode(self, xor_stream):
        # Initialize with BOM context
        context0 = int.from_bytes(BOM * 6, "big")
        beam = [{
            'ctx1': context0,
            'ctx2': context0,
            'log_prob': 0.0,
            'path': bytearray(),
            'back_ptr': None  # For path reconstruction
        }]

        # Store beam history for reconstruction
        beam_history = []

        for xor_byte in xor_stream:
            new_beam = []
            for state in beam:
                for p1 in range(256):
                    p2 = p1 ^ xor_byte

                    # Get probabilities
                    ctx1_bytes = state['ctx1'].to_bytes(6, 'big')
                    ctx2_bytes = state['ctx2'].to_bytes(6, 'big')
                    log_prob1 = self.model1.log_prob(p1, ctx1_bytes)
                    log_prob2 = self.model2.log_prob(p2, ctx2_bytes)

                    # Skip impossible combinations
                    if log_prob1 < -20 or log_prob2 < -20:
                        continue

                    # Create new state
                    new_ctx1 = ((state['ctx1'] << 8) | p1) & self.context_mask
                    new_ctx2 = ((state['ctx2'] << 8) | p2) & self.context_mask
                    new_log_prob = state['log_prob'] + log_prob1 + log_prob2

                    new_beam.append({
                        'ctx1': new_ctx1,
                        'ctx2': new_ctx2,
                        'log_prob': new_log_prob,
                        'path': state['path'] + bytes([p1, p2]),
                        'back_ptr': state
                    })

            # Prune beam
            new_beam.sort(key=lambda x: x['log_prob'], reverse=True)
            beam = new_beam[:self.beam_width]
            beam_history.append(beam)

        # Reconstruct best path
        best_state = max(beam, key=lambda x: x['log_prob'])
        p1 = bytearray()
        p2 = bytearray()

        state = best_state
        while state['back_ptr']:
            path = state['path'][-2:]  # Last two bytes
            p1.append(path[0])
            p2.append(path[1])
            state = state['back_ptr']

        return bytes(p1)[::-1], bytes(p2)[::-1]  # Reverse to original order