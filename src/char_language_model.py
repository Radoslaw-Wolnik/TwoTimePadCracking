import math
from collections import defaultdict

BOM = b'\x01'  # Start-of-heading character
EOM = b'\x02'   # Start-of-text character


class CharLanguageModel:
    def __init__(self, n=7):
        self.n = n
        self.ngrams = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.backoff_weights = {}
        self.prob_cache = {}

    def train(self, text):
        for i in range(len(text) - self.n):
            context = text[i:i + self.n - 1]
            char = text[i + self.n - 1]
            self.ngrams[context][char] += 1
            self.context_counts[context] += 1
            self.vocab.add(char)

            self._compute_witten_bell_weights()

    def _compute_witten_bell_weights(self):
        for context in self.context_counts:
            # Number of distinct continuations
            types = len(self.ngram_counts[context])
            total = self.context_counts[context]
            # λ = 1 - (types / (types + total))
            self.backoff_weights[context] = 1 - types / (types + total)

    def log_prob(self, char, context):
        # Base case for empty context
        if len(context) == 0:
            return math.log(1 / self.vocab_size)

        # Get probability with full context
        full_prob = self.ngram_counts[context].get(char, 0) / self.context_counts[context]

        # Apply Witten-Bell smoothing
        if full_prob > 0:
            return math.log(full_prob)
        else:
            # Back off to shorter context
            backoff_weight = self.backoff_weights[context]
            shorter_context = context[1:]
            backoff_prob = math.exp(self.log_prob(char, shorter_context))
            return math.log(backoff_weight * backoff_prob)

    def get_prob_table(self, max_contexts=1000000):
        """Create probability table for most common contexts"""
        # Sort contexts by frequency
        sorted_contexts = sorted(
            self.context_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_contexts]

        # Create probability table
        table = np.zeros((max_contexts, 256), dtype=np.float32)
        table.fill(1e-6)  # Default low probability

        for idx, (context, _) in enumerate(sorted_contexts):
            for char in range(256):
                # Get probability with smoothing
                prob = self.log_prob(char, context)
                table[idx, char] = np.exp(prob)

        return table