# __init__.py -- import-friendly wrapper with numpy fallback
try:
    from ._twotimepad import compute_combined_log_probs # compiled extension
    CYTHON_OK = True
except Exception: # pragma: no cover - fallback
    CYTHON_OK = False
    import numpy as np

    def compute_combined_log_probs(ctx1, ctx2, xor_byte, table1, table2, table_size):
        """Pure-numpy fallback: returns array of combined log-probs length 256."""
        ts = int(table_size)
        t1 = np.asarray(table1, dtype=np.float64)
        t2 = np.asarray(table2, dtype=np.float64)
        base1 = (int(ctx1) % ts) * 256
        base2 = (int(ctx2) % ts) * 256
        idxs1 = base1 + np.arange(256)
        idxs2 = base2 + (np.arange(256) ^ int(xor_byte))
        vals1 = t1[idxs1]
        vals2 = t2[idxs2]
        out = np.empty(256, dtype=np.float64)
        mask = (vals1 > 0) & (vals2 > 0)
        out[~mask] = -1e300
        out[mask] = np.log(vals1[mask]) + np.log(vals2[mask])
        return out