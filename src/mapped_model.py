import mmap
import os
import struct
import numpy as np
from collections import defaultdict
from . import CharLanguageModel, parse_emails, preprocess_email

class MappedLanguageModel:
    def __init__(self, model_path):
        self.file = open(model_path, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), 0)
        header = struct.unpack('II', self.mmap[:8])
        self.n = header[0]
        self.vocab_size = header[1]
        self.offset = 8

        # Build index for contexts
        self.index = {}
        pos = self.offset
        while pos < len(self.mmap):
            ctx_hash = struct.unpack('I', self.mmap[pos:pos + 4])[0]
            self.index[ctx_hash] = pos + 4
            pos += 4 + 256 * 8  # Move to next context

    def log_prob(self, char, context):
        ctx_hash = self._hash_context(context)
        if ctx_hash not in self.index:
            # Backoff to shorter context
            return self.log_prob(char, context[1:]) if context else np.log(1 / self.vocab_size)

        pos = self.index[ctx_hash]
        prob = struct.unpack('d', self.mmap[pos + char * 8:pos + (char + 1) * 8])[0]
        return np.log(prob)

    def _hash_context(self, context):
        # Simple rolling hash
        h = 0
        for c in context:
            h = (h * 31 + c) & 0xFFFFFFFF
        return h

    @classmethod
    def build(cls, ngram_model, output_path):
        with open(output_path, 'wb') as f:
            # Write header: n, vocab_size
            f.write(struct.pack('II', ngram_model.n, len(ngram_model.vocab)))

            # Write each context
            for context, probs in ngram_model.ngram_probs.items():
                ctx_hash = cls._static_hash_context(context)
                f.write(struct.pack('I', ctx_hash))

                # Write probabilities for all 256 bytes
                for char in range(256):
                    prob = probs.get(char, 0)
                    f.write(struct.pack('d', prob))

    def load_or_train_model(corpus_path, model_path, use_mmap=True):
        if os.path.exists(model_path + ".mmap"):
            return MappedLanguageModel(model_path + ".mmap")

        print("Training new language model...")
        model = CharLanguageModel(n=7)

        # Train on email corpus
        email_count = 0
        for email_content in parse_emails(corpus_path):
            text = preprocess_email(email_content)
            model.train(text)
            email_count += 1
            if email_count % 1000 == 0:
                print(f"Processed {email_count} emails")

        # Save and build memory-mapped version
        model.save(model_path)
        if use_mmap:
            MappedLanguageModel.build(model, model_path + ".mmap")
            return MappedLanguageModel(model_path + ".mmap")
        return model