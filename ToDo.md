## Running 

`python -m src.train_model --corpus_type email --corpus_path processed_emails --model_path email_model.bin`

`python -m src.train_model --corpus_type email --corpus_path processed_emails --model_path email_model.bin --download`

`python -m src.quick_test`

`python -m src.test_model`

`python -m src.run_complete_test`

## Final Notes

### What's Complete
- Core language model with Witten-Bell smoothing
- Efficient decoder with beam search
- Memory-mapped model storage
- Email preprocessing pipeline
- Evaluation metrics from paper
- Cython acceleration

### Next Steps
1. Implement the Enron download script
2. Find/create test encrypted email pairs
3. Add unit tests for all components
4. Experiment with different n-gram sizes
5. Optimize beam width for email content

The implementation now closely follows the paper's approach while focusing on email cryptanalysis. 
The memory-mapped models and Cython acceleration enable handling real-world datasets while maintaining the performance characteristics described in the paper.



Temp - previous cython code
```cython
# distutils: language=c++
# cython: boundscheck=False, wraparound=False

import cython
from libcpp.vector cimport vector
from libcpp.pair cimport pair
import numpy as np
cimport numpy as np

cdef struct State:
    int context1
    int context2
    double log_prob

cdef vector[State] process_byte(
        vector[State] beam,
        unsigned char xor_byte,
        double * prob_table1,
        double * prob_table2,
        int beam_width
):
    cdef vector[State] new_beam
    cdef State new_state
    cdef unsigned char c, p1, p2
    cdef double prob1, prob2, new_prob

    cdef int ctx1, ctx2

    for i in range(beam.size()):
        state = beam[i]
        for c in range(256):
            p1 = c
            p2 = c ^ xor_byte

            # Get probabilities from precomputed tables
            prob1 = prob_table1[state.context1 * 256 + p1]
            prob2 = prob_table2[state.context2 * 256 + p2]
            new_prob = state.log_prob + prob1 + prob2

            # Update contexts (simple shift register)
            new_state.context1 = (state.context1 * 256 + p1) % (256 ** (6))  # Keep last 6 chars
            new_state.context2 = (state.context2 * 256 + p2) % (256 ** (6))
            new_state.log_prob = new_prob

            new_beam.push_back(new_state)

    # Prune beam
    sort(new_beam.begin(), new_beam.end(), compare_states)
    if new_beam.size() > beam_width:
        new_beam.resize(beam_width)
    return new_beam
```