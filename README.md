# Automated Cryptanalysis of Two-Time Pads

## Overview

This project implements the complete cryptanalysis system from the 2006 paper [**"A Natural Language Approach to Automated Cryptanalysis of Two-time Pads"**](https://www.cs.jhu.edu/~jason/papers/mason+al.ccs06.pdf) by Mason, Watkins, Eisner, and Stubblefield. 

The system demonstrates a practical attack on **keystream reuse vulnerabilities**, automatically recovering plaintexts from two ciphertexts encrypted with the same keystream when only their document types are known (e.g., English emails, HTML pages, Word documents).

## Core Cryptographic Problem

### The Two-Time Pad Vulnerability
When a keystream `k` is reused to encrypt two plaintexts `p` and `q`, an attacker can compute:

```
c₁ = p ⊕ k  (ciphertext 1)
c₂ = q ⊕ k  (ciphertext 2)
x  = c₁ ⊕ c₂ = (p ⊕ k) ⊕ (q ⊕ k) = p ⊕ q
```

The challenge is to recover `p` and `q` from `x = p ⊕ q` alone. This has 2ⁿ solutions for n-bit messages, but when plaintexts follow known statistical patterns (natural language), the solution becomes tractable.

### Language Model Approach
The paper's key insight: treat plaintext recovery as a **decoding problem** using statistical language models. Given `x = p ⊕ q` and language models for `p` and `q`'s document types, find the most probable `(p, q)` pair:

```
argmax_(p,q) Pr₁(p) · Pr₂(q)   subject to   p ⊕ q = x
```

Where Pr₁ and Pr₂ are character-level n-gram language models trained on domain-specific corpora.

## Character-Level N-Gram Language Model

### What Is a Character N-Gram Model?
A **character n-gram model** is a probability distribution over the next character given the previous `n-1` characters. For `n=7` (used in this implementation), the model predicts the 7th character based on the preceding 6 characters.

**Example**: After seeing "hobnob", the model might predict:
- 's' with probability 0.52 (forming "hobnobs")
- space with probability 0.35 (forming "hobnob ")
- 'b' with probability 0.13 (forming "hobnobb")

### Mathematical Formulation
For a plaintext string `p = (p₁, p₂, ..., pₗ)`, the 7-gram model defines:

```
Pr₁(p) = ∏ᵢ₌₁ˡ Pr₁(pᵢ | pᵢ₋₆, ..., pᵢ₋₁)
```

Each factor represents the probability of character `pᵢ` given the previous 6 characters, assuming a **Markov property** - each character depends only on the immediate history.

### Training Process

The model is trained by sliding a 7-character window through training texts and counting occurrences:

```python
def train(self, text: bytes):
    # Add BOM (beginning-of-message) padding for proper context
    prefix = bytes([BOM]) * 6  # 6 BOM characters
    suffix = bytes([EOM])      # 1 EOM character
    data = prefix + text + suffix
    
    # Count 7-grams: (context of 6 chars) → (next char)
    for i in range(0, len(data) - 7 + 1):
        context = data[i:i+6]      # 6-byte context
        next_char = data[i+6]      # Character to predict
        self.ngram_counts[context][next_char] += 1
```

**Example**: Training on "hobnobs" with BOM padding:
```
Training window: "h o b n o b s"
Context: "hobnob" (6 chars)
Next char: 's'
Result: ngram_counts[b"hobnob"]['s'] += 1
```

### The Zero-Probability Problem & Witten-Bell Smoothing

#### The Problem
With 6-character contexts, there are 256⁶ ≈ 2.8×10¹⁴ possible contexts. Even with billions of training characters, most contexts are never seen. Naive frequency estimates give zero probability for unseen sequences, making decoding impossible.

#### Witten-Bell Solution
Witten-Bell smoothing **interpolates** between:
1. **Maximum Likelihood Estimate (MLE)** for observed n-grams
2. **Backoff estimate** using shorter contexts

For a context `c`:
- `N(c)` = total times `c` appeared
- `T(c)` = distinct characters following `c`
- **Backoff weight**: `λ(c) = T(c) / (T(c) + N(c))`

**Probability calculation**:
```
P_WB(x|c) = (1 - λ(c)) × P_ML(x|c) + λ(c) × P_backoff(x|c')
```
Where `c' = c[1:]` (context with oldest character removed)

#### Recursive Backoff Chain
When computing `P('s' | "hobnob")`:
1. Try 7-gram: `P('s' | "hobnob")`
2. If insufficient data, backoff to 6-gram: `P('s' | "obnob")`
3. Continue backing off until 1-gram: `P('s')`
4. Base case: uniform over vocabulary

This ensures **no probability is ever zero**, allowing the decoder to explore all possibilities.

## Decoding Algorithm: Beam Search with Viterbi

### Problem Formalization
Given `x = p ⊕ q` and language models for `p` and `q`, find:
```
argmax_(p,q) ∏ᵢ Pr₁(pᵢ | pᵢ₋₆...pᵢ₋₁) × Pr₂(qᵢ | qᵢ₋₆...qᵢ₋₁)
subject to pᵢ ⊕ qᵢ = xᵢ for all i
```

### Beam State Representation

```python
class BeamState:
    __slots__ = ['ctx1', 'ctx2', 'log_prob', 'p1_path', 'p2_path']
    
    def __init__(self, ctx1: bytes, ctx2: bytes, log_prob: float,
                 p1_path: bytearray, p2_path: bytearray):
        self.ctx1 = ctx1  # Last 6 chars of plaintext1
        self.ctx2 = ctx2  # Last 6 chars of plaintext2
        self.log_prob = log_prob  # Cumulative log probability
        self.p1_path = p1_path    # Reconstructed plaintext1 so far
        self.p2_path = p2_path    # Reconstructed plaintext2 so far
```

### Beam Search Algorithm

```python
def decode(self, xor_stream: List[int]) -> Tuple[bytes, bytes]:
    # Initialize with BOM contexts
    beam = [BeamState(bytes([BOM]*6), bytes([BOM]*6), 0.0, bytearray(), bytearray())]
    
    for i, xor_byte in enumerate(xor_stream):
        candidates = []
        
        for state in beam:
            # Generate candidate byte pairs (prefer printable ASCII)
            for p1_byte in range(32, 127):  # Printable chars
                p2_byte = p1_byte ^ xor_byte
                
                if 32 <= p2_byte <= 126:  # Both printable
                    # Score using language models
                    lp1 = model1.log_prob(p1_byte, state.ctx1)
                    lp2 = model2.log_prob(p2_byte, state.ctx2)
                    new_prob = state.log_prob + lp1 + lp2
                    
                    # Update contexts
                    new_ctx1 = (state.ctx1 + bytes([p1_byte]))[-6:]
                    new_ctx2 = (state.ctx2 + bytes([p2_byte]))[-6:]
                    
                    candidates.append(BeamState(new_ctx1, new_ctx2, new_prob,
                                                state.p1_path + [p1_byte],
                                                state.p2_path + [p2_byte]))
        
        # Keep only top-k states (beam width = 100)
        beam = sorted(candidates, key=lambda s: s.log_prob, reverse=True)[:100]
    
    # Return best candidate
    best = max(beam, key=lambda s: s.log_prob)
    return bytes(best.p1_path), bytes(best.p2_path)
```

### Computational Complexity
- **Full search space**: O(256²ˡ) for length `l` (intractable)
- **With n-gram constraint**: O(l × 256⁷) (still massive)
- **With beam search**: O(l × beam_width × 256) (practical)

**Performance**: ~200ms per byte, matching the paper's results on a $2,000 PC.

## Experimental Results

We conducted extensive experiments following the paper's methodology, training models on Enron emails of varying sizes and evaluating recovery accuracy.

### Methodology
- **Training**: Character-level 7-gram models with Witten-Bell smoothing
- **Testing**: 80/20 train/test split with cross-validation
- **Beam width**: 100 states
- **Evaluation metrics**: Byte accuracy, pair accuracy, switch rate

### Results Summary

| Training Emails | Byte Accuracy | Pair Accuracy | Switch Rate | Success Rate* |
|----------------|---------------|---------------|-------------|--------------|
| 1,250 | 33.4% ± 35.5% | 63.6% ± 34.3% | 30.2% | 50% |
| 6,250 | 42.5% ± 39.9% | 70.4% ± 32.6% | 27.9% | 60% |
| 12,500 | 36.6% ± 37.7% | 79.3% ± 27.5% | 42.8% | 66% |

*Success rate = percentage of tests with pair accuracy > 70%

### Key Insights

1. **Pair Accuracy > Byte Accuracy**: The significant gap indicates the **"switching streams" problem** - bytes are correctly recovered but assigned to wrong texts.

2. **Diminishing Returns**: Byte accuracy doesn't monotonically increase with more training data, suggesting diversity matters more than quantity.

3. **High Variance**: Large standard deviations indicate performance depends heavily on specific email content.

4. **Switch Rate Correlates with Gap**: Higher switch rates correspond to larger differences between pair and byte accuracy.

## The "Switching Streams" Problem

### Problem Description
After decoding errors, the algorithm loses track of which byte belongs to which text. Correct bytes continue to be recovered but are assigned to the wrong plaintext until another correction occurs.

**Example**:
```
Original:    p = "Hello world", q = "Secret message"
Recovered:   p' = "Secret Hello", q' = "world message"
Pair Accuracy: 100% (all bytes present)
Byte Accuracy: 50% (bytes in wrong positions)
```

### Why It Happens
Character n-gram models only look at the last 6 characters. When the decoder makes an error, the contexts become corrupted. Once it "gets back on track," it has no memory of which stream the correct bytes belong to.

### Solutions from the Paper

#### 1. Multiple Keystream Reuse
If `k` encrypts three texts `p, q, r`, we get two constraints:
```
x₁ = p ⊕ q
x₂ = p ⊕ r
```
The extra constraint prevents arbitrary swapping since any assignment must be consistent across all three texts.

#### 2. Different Document Types
When `p` is HTML and `q` is email, their language models differ significantly:
- HTML: High probability for `<`, `>`, tags
- Email: High probability for `@`, `:` in headers
The probability gap makes swapped assignments highly improbable.

#### 3. Post-Processing Detection
```python
def detect_and_fix_switches(rec1, rec2, model1, model2):
    """
    Detect segments where bytes are likely swapped
    """
    for i in range(len(rec1)):
        # Compute probability of current assignment
        prob_current = model1.log_prob(rec1[i], context1) + \
                      model2.log_prob(rec2[i], context2)
        
        # Compute probability if swapped
        prob_swapped = model1.log_prob(rec2[i], context1) + \
                      model2.log_prob(rec1[i], context2)
        
        # Mark for swapping if swapped has higher probability
        if prob_swapped > prob_current + threshold:
            mark_swap_position(i)
```

## Running the Project

### Installation
```bash
git clone https://github.com/Radoslaw-Wolnik/TwoTimePadCracking.git
cd two-time-pad-decrypt
pip install -r requirements.txt

# Build Cython extensions (optional, for performance)
cd src
python setup.py build_ext --inplace
```

### Basic Usage

#### 1. Setup Data
```bash
# Download and process Enron emails
python -m src.main --setup --type email

# For HTML documents
python -m src.main --setup --type html

# For Word documents
python -m src.main --setup --type word
```

#### 2. Train Language Models
```bash
# Train a new email model
python -m src.main --train --type email --model-name email_model --new

# Continue training on existing model
python -m src.main --train --type email --model-name email_model --open-model-name email_model
```

#### 3. Perform Cryptanalysis
```bash
# Decode two ciphertexts encrypted with same keystream
python -m src.main --decoding \
    --model-name email_model \
    --doc-type email \
    --file1path enc1.bin \
    --file2path enc2.bin
```

#### 4. Run Analysis Experiments
```bash
# Analyze with different training sizes
python analysis.py --num-emails 1000 --num-runs 3 --num-tests 10
python analysis.py --num-emails 5000 --num-runs 3 --num-tests 10
python analysis.py --num-emails 10000 --num-runs 3 --num-tests 10

# With custom parameters
python analysis.py --num-emails 5000 --n 7 --beam-width 200 --num-runs 5
```

#### 5. Run Tests
```bash
# All tests
python -m pytest src/tests/ -v

# Specific test suites
python -m pytest src/tests/unit/test_char_language_model.py -v
python -m pytest src/tests/integration/test_decoder_integration.py -v

# Slow tests (requires data)
python -m pytest src/tests/ -m slow -v
```

### Expected Output
- **Models**: Saved to `res/models/` directory as binary files
- **Analysis Results**: JSON, plots, and summaries in `res/analysis_results/`
- **Decoded Text**: Recovered plaintexts printed to console and saved to files

## References & Citations

1. **Primary Reference**: Mason, J., Watkins, K., Eisner, J., & Stubblefield, A. (2006). *A Natural Language Approach to Automated Cryptanalysis of Two-time Pads*. Proceedings of the 13th ACM Conference on Computer and Communications Security (CCS'06).

2. **Language Modeling**: Witten, I. H., & Bell, T. C. (1991). *The zero-frequency problem: Estimating the probabilities of novel events in adaptive text compression*. IEEE Transactions on Information Theory.

3. **Earlier Work**: Dawson, E., & Nielsen, L. (1996). *Automated cryptanalysis of XOR plaintext strings*. Cryptologia.

4. **Historical Context**: Kahn, D. (1996). *The Codebreakers*. Comprehensive history of cryptography including two-time pad vulnerabilities.

5. **Dataset**: Enron Email Dataset, available at https://www.cs.cmu.edu/~enron/

## License

This project is for **educational and research purposes only**. Use responsibly and in accordance with applicable laws and regulations.

The code is provided as-is under the MIT License. See LICENSE file for details.

## Responsible Disclosure

This project demonstrates a known cryptographic vulnerability (keystream reuse). Use this knowledge responsibly:
- Only test on systems you own or have explicit permission to test
- Never use for unauthorized access to systems or data
- Report security vulnerabilities to vendors through proper channels

---

**Author**: Radoslaw Wolnik  
**Contact**: radoslaw.m.wolnik@gmail.com  
**Repository**: [GitHub Link](https://github.com/Radoslaw-Wolnik/TwoTimePadCracking)\
**Based on**: Mason et al. (2006), CCS'06
