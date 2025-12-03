# src/model/char_language_model.py
from __future__ import annotations
from pathlib import Path
from collections import defaultdict, Counter
import math
import struct
import tempfile
import os
from typing import Dict, DefaultDict, Iterable, Union

BOM = 0x01  # boundary marker as integer
EOM = 0x02


class CharLanguageModel:
    def __init__(self, n: int = 7) -> None:
        if n < 2:
            raise ValueError("n must be >= 2 (context length = n-1)")
        self.n = n
        self.context_length = n - 1
        self.ngram_counts: DefaultDict[bytes, Counter] = defaultdict(Counter)
        self.context_counts: Dict[bytes, int] = {}
        self.vocab: set[int] = set()
        self.backoff_weights: Dict[bytes, float] = {}
        self.min_prob = 1e-12

    def train(self, text: Union[str, bytes, Iterable[int]]) -> None:
        """Train the model on text. Prepend BOMs so initial bytes become
        predictable and append an EOM so endings are modeled."""
        if isinstance(text, str):
            text_bytes = text.encode("utf-8", "ignore")
        elif isinstance(text, bytes):
            text_bytes = text
        else:
            text_bytes = bytes(text)

        if len(text_bytes) == 0:
            return

        # add BOM padding at the start so first bytes show up as next_char
        prefix = bytes([BOM]) * self.context_length
        # append a single EOM so final contexts are observed
        suffix = bytes([EOM])
        data = prefix + text_bytes + suffix

        L = len(data)
        if L < self.n:
            return

        # Slide window of size n
        for i in range(0, L - self.n + 1):
            context = data[i:i + self.context_length]  # n-1 bytes
            next_char = data[i + self.context_length]  # single byte
            self.ngram_counts[context][next_char] += 1

        # Fill context_counts and vocab
        for ctx, counter in self.ngram_counts.items():
            self.context_counts[ctx] = sum(counter.values())
            self.vocab.update(counter.keys())

        self._compute_witten_bell_weights()

    def _compute_witten_bell_weights(self) -> None:
        """Compute Witten-Bell smoothing weights for each context.

        We store backoff weight = T / (N + T), where:
          - T = number of distinct continuation symbols after context
          - N = total number of continuation tokens after context
        """
        for ctx, counter in self.ngram_counts.items():
            total_count = self.context_counts.get(ctx, 0)
            distinct_chars = len(counter)

            if total_count + distinct_chars > 0:
                self.backoff_weights[ctx] = distinct_chars / (distinct_chars + total_count)
            else:
                self.backoff_weights[ctx] = 0.0

    def log_prob(self, char: int, context: bytes) -> float:
        """Return log probability for single byte given context using Witten-Bell smoothing."""
        if not isinstance(char, int) or not (0 <= char <= 255):
            raise ValueError("char must be integer in [0,255]")

        # Get appropriate context length (take last context_length bytes)
        ctx = context[-self.context_length:] if context else b''

        # Base case: no context -> only give mass to observed vocabulary
        if len(ctx) == 0:
            if not self.vocab:
                return math.log(self.min_prob)
            # only observed chars get uniform probability at the top level;
            # unseen chars get a tiny probability.
            if char in self.vocab:
                vocab_size = max(1, len(self.vocab))
                return math.log(1.0 / vocab_size)
            else:
                return math.log(self.min_prob)

        # If we have counts for this context
        if ctx in self.ngram_counts:
            total_count = self.context_counts.get(ctx, 0)
            char_count = self.ngram_counts[ctx].get(char, 0)

            backoff_weight = self.backoff_weights.get(ctx, 0.0)
            shorter_ctx = ctx[1:] if len(ctx) > 1 else b''
            backoff_prob = math.exp(self.log_prob(char, shorter_ctx))

            if char_count > 0:
                # Interpolate MLE with backoff (Witten-Bell):
                #  P = (N/(N+T)) * (c / N) + (T/(N+T)) * P_shorter
                # Here backoff_weight = T/(N+T)
                mle = char_count / total_count
                mle_weight = 1.0 - backoff_weight
                prob = mle * mle_weight + backoff_weight * backoff_prob
                return math.log(max(prob, self.min_prob))
            else:
                # Character not seen in this context -> use backoff term only
                prob = backoff_weight * backoff_prob
                return math.log(max(prob, self.min_prob))
        else:
            # Context not found - backoff immediately to shorter context
            shorter_ctx = ctx[1:] if len(ctx) > 1 else b''
            return self.log_prob(char, shorter_ctx)

    # ----- Serialization (binary) -----
    # Format (all little-endian '<'):
    # 4 bytes magic b'CLM1'
    # 4 bytes version (I) = 1
    # 4 bytes n (I)
    # 4 bytes vocab_size (I)
    # 4 bytes num_contexts (I)
    # Then for each context:
    #   context (n-1 bytes)
    #   total_count (I)
    #   num_chars (I)
    #   for each char: (B I) -> (1byte, 4byte count)
    def save(self, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp = tempfile.mkstemp(prefix="clm_", suffix=".bin", dir=str(file_path.parent))
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                f.write(b'CLM1')
                f.write(struct.pack('<I', 1))
                f.write(struct.pack('<I', self.n))
                f.write(struct.pack('<I', len(self.vocab)))
                f.write(struct.pack('<I', len(self.ngram_counts)))

                for ctx, counter in self.ngram_counts.items():
                    if len(ctx) != self.n - 1:
                        raise ValueError("Context length mismatch")
                    f.write(ctx)
                    total = self.context_counts.get(ctx, sum(counter.values()))
                    f.write(struct.pack('<I', total))
                    f.write(struct.pack('<I', len(counter)))
                    for ch, cnt in counter.items():
                        f.write(struct.pack('<B', ch))
                        f.write(struct.pack('<I', cnt))

            os.replace(tmp, str(file_path))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "CharLanguageModel":
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with open(file_path, "rb") as f:
            magic = f.read(4)
            if magic != b'CLM1':
                raise IOError("Bad magic")
            version = struct.unpack('<I', f.read(4))[0]
            if version != 1:
                raise IOError("Unsupported version")
            n = struct.unpack('<I', f.read(4))[0]
            vocab_size = struct.unpack('<I', f.read(4))[0]   # unused for now
            num_contexts = struct.unpack('<I', f.read(4))[0]

            model = cls(n=n)
            for i in range(num_contexts):
                ctx = f.read(model.n - 1)
                if len(ctx) < (model.n - 1):
                    raise IOError("Unexpected EOF reading context")
                total = struct.unpack('<I', f.read(4))[0]
                num_chars = struct.unpack('<I', f.read(4))[0]
                counter = Counter()
                for _ in range(num_chars):
                    b = f.read(1)
                    if not b:
                        raise IOError("Unexpected EOF char byte")
                    ch = b[0]
                    cnt = struct.unpack('<I', f.read(4))[0]
                    counter[ch] = cnt
                    model.vocab.add(ch)
                model.ngram_counts[ctx] = counter
                model.context_counts[ctx] = total

        model._compute_witten_bell_weights()
        return model
