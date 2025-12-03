# src/model/char_language_model.py
from __future__ import annotations
from pathlib import Path
from collections import defaultdict, Counter
import math
import struct
import tempfile
import os
from typing import Dict, DefaultDict, Iterable, Union

BOM = b'\x01'  # boundary marker
EOM = b'\x02'

Context = bytes  # context is a fixed-length bytes object
Char = int       # single byte as int in [0,255]


class CharLanguageModel:
    def __init__(self, n: int = 7) -> None:
        if n < 2:
            raise ValueError("n must be >= 2 (context length = n-1)")
        self.n = n
        # mapping: context (bytes) -> Counter of next-char (int -> count)
        self.ngram_counts: DefaultDict[Context, Counter] = defaultdict(Counter)
        self.context_counts: Dict[Context, int] = {}
        self.vocab: set[int] = set()
        self.backoff_weights: Dict[Context, float] = {}
        self.min_prob = 1e-12

    def train(self, text: Union[str, bytes, Iterable[int]]) -> None:
        """Train the model on a text (str or bytes). Handles boundaries."""
        if isinstance(text, str):
            text_bytes = text.encode("utf-8", "ignore")
        elif isinstance(text, bytes):
            text_bytes = text
        else:
            # assume iterable of ints
            text_bytes = bytes(text)

        # Add boundary markers
        data = BOM + text_bytes + EOM
        L = len(data)
        if L < self.n:
            return

        # Slide window
        for i in range(0, L - self.n + 1):
            context = data[i:i + self.n - 1]           # n-1 bytes
            next_char = data[i + self.n - 1]           # single byte (int via indexing)
            # store as int for counters
            self.ngram_counts[context][next_char] += 1

        # fill context_counts and vocab
        for ctx, counter in self.ngram_counts.items():
            self.context_counts[ctx] = sum(counter.values())
            self.vocab.update(counter.keys())

        self._compute_witten_bell_weights()

    def _compute_witten_bell_weights(self) -> None:
        for ctx, total_count in self.context_counts.items():
            distinct = len(self.ngram_counts[ctx])
            # Witten-Bell lambda: distinct / (distinct + total_count)
            self.backoff_weights[ctx] = distinct / (distinct + total_count) if (distinct + total_count) > 0 else 0.0

    def log_prob(self, char: Union[int, bytes], context: Union[bytes, bytearray]) -> float:
        """Return log probability for single byte `char` given `context` (bytes).
           Works recursively with backoff.
        """
        if isinstance(char, bytes):
            if len(char) != 1:
                raise ValueError("char bytes must be length 1")
            c = char[0]
        else:
            c = int(char)

        # ensure context is bytes and at most n-1 in length
        ctx = bytes(context)[- (self.n - 1):] if context else b''

        # base case: empty context
        if len(ctx) == 0:
            if len(self.vocab) == 0:
                return math.log(self.min_prob)
            return math.log(1.0 / max(1, len(self.vocab)))

        # If we have counts for this context
        if ctx in self.ngram_counts:
            cnt = self.ngram_counts[ctx].get(c, 0)
            if cnt > 0:
                return math.log(cnt / self.context_counts[ctx])

        # backoff
        backoff_weight = self.backoff_weights.get(ctx, 0.4)
        shorter_ctx = ctx[1:]  # drop first byte (leftmost) to shorten
        backoff_log = self.log_prob(c, shorter_ctx)
        backoff_prob = math.exp(backoff_log)

        smoothed = backoff_weight * backoff_prob + (1 - backoff_weight) * self.min_prob
        return math.log(max(smoothed, self.min_prob))

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
