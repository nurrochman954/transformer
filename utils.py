# utils.py
# utilitas: softmax stabil dan pembuatan causal mask
import numpy as np

def softmax_stable(x, axis=-1):
    """Softmax numerik stabil.
    x: array numpy
    axis: dimensi softmax
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def causal_mask(seq_len):
    """Buat additive mask untuk mencegah perhatian ke token masa depan.
    Menghasilkan matriks shape (seq_len, seq_len) dengan 0 di area yang
    diizinkan dan -1e9 di area yang harus di-mask (masa depan).
    """
    # upper triangular (k=1) -> True untuk elemen di atas diagonal (masa depan)
    mask_bool = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    # return additive mask
    return np.where(mask_bool, -1e9, 0.0)
