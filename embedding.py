# embedding.py
# TokenEmbedding dan PositionalEncoding (sinusoidal / learned)
import numpy as np

class TokenEmbedding:
    """Embedding token sederhana.
    - vocab_size: jumlah token di vocabulary
    - d_model: dimensi embedding
    """
    def __init__(self, vocab_size, d_model, rng=None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        if rng is None:
            rng = np.random.default_rng(0)
        # inisialisasi embedding (skala kecil)
        self.W = rng.normal(scale=(d_model ** -0.5), size=(vocab_size, d_model))

    def __call__(self, token_ids):
        """token_ids: array int shape (batch, seq_len)
        kembalian: embeddings shape (batch, seq_len, d_model)
        """
        return self.W[token_ids]

class PositionalEncoding:
    """Positional encoding: metode 'sinusoidal' atau 'learned'.
    - d_model: dim embedding
    - max_len: panjang maksimal posisi
    - method: 'sinusoidal' atau 'learned'
    """
    def __init__(self, d_model, max_len=512, method='sinusoidal', rng=None):
        self.d_model = d_model
        self.max_len = max_len
        self.method = method
        if rng is None:
            rng = np.random.default_rng(0)

        if method == 'sinusoidal':
            # buat sinusoidal positional encoding (Vaswani et al.)
            pe = np.zeros((max_len, d_model), dtype=float)
            position = np.arange(0, max_len)[:, np.newaxis]  # (max_len, 1)
            div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)
            self.pe = pe  # (max_len, d_model)
        elif method == 'learned':
            # learned positional embedding (di-train kalau ada training)
            self.pe = rng.normal(scale=(d_model ** -0.5), size=(max_len, d_model))
        else:
            raise ValueError("method harus 'sinusoidal' atau 'learned'")

    def __call__(self, seq_len):
        """Kembalikan positional encoding untuk panjang seq_len"""
        if seq_len > self.max_len:
            raise ValueError("seq_len melebihi max_len positional encoding")
        return self.pe[:seq_len]  # shape (seq_len, d_model)
