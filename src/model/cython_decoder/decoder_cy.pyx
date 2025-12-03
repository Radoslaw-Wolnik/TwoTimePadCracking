# cython: boundscheck=False, wraparound=False, cdivision=True
# _twotimepad.pyx

from libc.stdint cimport uint8_t, uint64_t
cimport numpy as np
import numpy as np


# compute_combined_log_probs:
# Given two large probability tables (flattened as table_size * 256),
# compute log_prob(p1) + log_prob(p2) for all p1=0..255 where p2 = p1 ^ xor_byte.
# Returns a numpy.ndarray[np.float64_t, ndim=1] length 256.


def compute_combined_log_probs(uint64_t ctx1, uint64_t ctx2, uint8_t xor_byte,
    np.ndarray[np.float64_t, ndim=1] table1,
    np.ndarray[np.float64_t, ndim=1] table2,
    int table_size):
    cdef:
        int i
        int base1, base2
        int idx1, idx2
        double v1, v2
        np.ndarray[np.float64_t, ndim=1] out = np.empty(256, dtype=np.float64)
        int ts = table_size


    # typed memoryviews for speed
    cdef double[:] t1 = table1
    cdef double[:] t2 = table2


    # Precompute bases
    base1 = (ctx1 % ts) * 256
    base2 = (ctx2 % ts) * 256


    for i in range(256):
        idx1 = base1 + i
        idx2 = base2 + (i ^ xor_byte)
        v1 = t1[idx1]
        v2 = t2[idx2]
        if v1 <= 0.0 or v2 <= 0.0:
            out[i] = -1e300 # effectively negative infinity (log(0))
        else:
            out[i] = np.log(v1) + np.log(v2)


    return out