# test_data.py
"""
Simple tests data for two-time pad decoding
"""

# Two short English texts
TEXT_A = b"This is a sample email message that contains some typical English text with common words and structure."
TEXT_B = b"Another example of text that might be found in documents, this one slightly different but still English."

# Create encrypted versions (simulating two-time pad)
KEY = b"\x42" * 100  # Simple repeating key for testing

ENCRYPTED_A = bytes(a ^ k for a, k in zip(TEXT_A, KEY))
ENCRYPTED_B = bytes(b ^ k for b, k in zip(TEXT_B, KEY))

# XOR stream (what an attacker would see)
XOR_STREAM = bytes(a ^ b for a, b in zip(ENCRYPTED_A, ENCRYPTED_B))

print(f"Text A length: {len(TEXT_A)}")
print(f"Text B length: {len(TEXT_B)}")
print(f"XOR stream: {XOR_STREAM[:20]}...")