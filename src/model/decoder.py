# src/decoder/two_time_pad_decoder.py
from typing import Tuple, List, Dict, Any
import math
import logging
from collections import defaultdict
from src.model.char_language_model import BOM, EOM

logger = logging.getLogger(__name__)


class BeamState:
    """Represents a state in the beam search."""
    __slots__ = ['ctx1', 'ctx2', 'log_prob', 'p1_path', 'p2_path', 'step']

    def __init__(self, ctx1: bytes, ctx2: bytes, log_prob: float,
                 p1_path: bytearray, p2_path: bytearray, step: int = 0):
        self.ctx1 = ctx1  # Last n-1 bytes of plaintext1
        self.ctx2 = ctx2  # Last n-1 bytes of plaintext2
        self.log_prob = log_prob
        self.p1_path = p1_path
        self.p2_path = p2_path
        self.step = step  # Current position


class TwoTimePadDecoder:
    def __init__(self, model1, model2, beam_width: int = 100, n: int = 7):
        self.model1 = model1
        self.model2 = model2
        self.beam_width = beam_width
        self.context_length = n - 1
        self.n = n

    def decode(self, xor_stream: List[int]) -> Tuple[bytes, bytes]:
        # Start with BOM contexts as in the paper
        initial_ctx1 = bytes([BOM] * self.context_length)
        initial_ctx2 = bytes([BOM] * self.context_length)

        initial_state = BeamState(
            ctx1=initial_ctx1,
            ctx2=initial_ctx2,
            log_prob=0.0,
            p1_path=bytearray(),
            p2_path=bytearray(),
            step=0
        )

        beam = [initial_state]

        for i, xor_byte in enumerate(xor_stream):
            candidates = []

            for state in beam:
                # Try plausible byte pairs first (printable ASCII)
                for p1_byte in range(32, 127):  # Printable ASCII
                    p2_byte = p1_byte ^ xor_byte

                    # If p2 is also printable, it's more likely
                    if 32 <= p2_byte <= 126:
                        lp1 = self.model1.log_prob(p1_byte, state.ctx1)
                        lp2 = self.model2.log_prob(p2_byte, state.ctx2)

                        if lp1 > -50 and lp2 > -50:  # Less strict threshold
                            new_log_prob = state.log_prob + lp1 + lp2

                            # Update contexts properly
                            new_ctx1 = (state.ctx1 + bytes([p1_byte]))[-self.context_length:]
                            new_ctx2 = (state.ctx2 + bytes([p2_byte]))[-self.context_length:]

                            new_p1_path = state.p1_path.copy()
                            new_p2_path = state.p2_path.copy()
                            new_p1_path.append(p1_byte)
                            new_p2_path.append(p2_byte)

                            candidates.append(BeamState(
                                ctx1=new_ctx1,
                                ctx2=new_ctx2,
                                log_prob=new_log_prob,
                                p1_path=new_p1_path,
                                p2_path=new_p2_path,
                                step=i + 1
                            ))

            # If no printable candidates, fall back to all bytes
            if not candidates:
                for state in beam:
                    for p1_byte in range(256):
                        p2_byte = p1_byte ^ xor_byte

                        lp1 = self.model1.log_prob(p1_byte, state.ctx1)
                        lp2 = self.model2.log_prob(p2_byte, state.ctx2)

                        new_log_prob = state.log_prob + lp1 + lp2

                        new_ctx1 = (state.ctx1 + bytes([p1_byte]))[-self.context_length:]
                        new_ctx2 = (state.ctx2 + bytes([p2_byte]))[-self.context_length:]

                        new_p1_path = state.p1_path.copy()
                        new_p2_path = state.p2_path.copy()
                        new_p1_path.append(p1_byte)
                        new_p2_path.append(p2_byte)

                        candidates.append(BeamState(
                            ctx1=new_ctx1,
                            ctx2=new_ctx2,
                            log_prob=new_log_prob,
                            p1_path=new_p1_path,
                            p2_path=new_p2_path,
                            step=i + 1
                        ))

            # Group by context and keep best per context
            context_best = {}
            for cand in candidates:
                key = (cand.ctx1, cand.ctx2)
                if key not in context_best or cand.log_prob > context_best[key].log_prob:
                    context_best[key] = cand

            # Take top-k by probability
            beam = sorted(context_best.values(),
                          key=lambda s: s.log_prob,
                          reverse=True)[:self.beam_width]

            if not beam:
                # Emergency fallback
                p1_byte = 32  # space
                p2_byte = p1_byte ^ xor_byte
                beam = [BeamState(
                    ctx1=bytes([BOM] * self.context_length),
                    ctx2=bytes([BOM] * self.context_length),
                    log_prob=-1000.0,
                    p1_path=bytearray([p1_byte]),
                    p2_path=bytearray([p2_byte]),
                    step=i + 1
                )]

        # Return best candidate
        best = max(beam, key=lambda s: s.log_prob)
        return bytes(best.p1_path), bytes(best.p2_path)