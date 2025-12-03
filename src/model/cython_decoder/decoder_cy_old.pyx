# distutils: language=c++
# cython: boundscheck=False, wraparound=False

import cython
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
import numpy as np
cimport numpy as np

# Context state structure
cdef struct ContextState:
    unsigned long long context1
    unsigned long long context2
    double log_prob
    vector[pair[unsigned char, unsigned char]] path

cdef bool compare_states(const ContextState& a, const ContextState& b):
    return a.log_prob > b.log_prob

cdef vector[ContextState] process_byte_cy(
        current_state,
        unsigned char xor_byte,
        np.float32_t * table1,
        np.float32_t * table2,
        int table_size,
        unsigned long long context_mask
):
    cdef vector[ContextState] new_states
    cdef ContextState new_state
    cdef unsigned char c, p1, p2
    cdef double prob1, prob2
    cdef unsigned long long new_context1, new_context2
    cdef unsigned long long context1 = current_state['context1']
    cdef unsigned long long context2 = current_state['context2']
    cdef double current_prob = current_state['log_prob']

    # Extract path from Python to C++ vector
    cdef vector[pair[unsigned char, unsigned char]] current_path
    for p in current_state['path']:
        current_path.push_back((p[0], p[1]))

    for c in range(256):
        p1 = c
        p2 = c ^ xor_byte

        # Get probabilities from precomputed tables
        idx1 = (context1 % table_size) * 256 + p1
        idx2 = (context2 % table_size) * 256 + p2
        prob1 = table1[idx1]
        prob2 = table2[idx2]

        # Handle zero probabilities
        if prob1 <= 0 or prob2 <= 0:
            continue

        new_prob = current_prob + np.log(prob1) + np.log(prob2)

        # Update contexts (48-bit for 6-byte context)
        new_context1 = ((context1 << 8) | p1) & context_mask
        new_context2 = ((context2 << 8) | p2) & context_mask

        # Create new state
        new_state.context1 = new_context1
        new_state.context2 = new_context2
        new_state.log_prob = new_prob
        new_state.path = current_path
        new_state.path.push_back((p1, p2))

        new_states.push_back(new_state)

    # Sort by probability (highest first)
    sort(new_states.begin(), new_states.end(), compare_states)
    return new_states