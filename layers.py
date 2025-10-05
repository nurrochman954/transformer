import numpy as np
from utils import softmax_stable

# Layer Normalization
class LayerNorm:
    """Layer normalization sederhana dengan gamma dan beta.
    - d_model: jumlah fitur
    - eps: epsilon stabilitas
    """
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        # parameter learnable (bisa diupdate saat training)
        self.gamma = np.ones((d_model,), dtype=float)
        self.beta  = np.zeros((d_model,), dtype=float)

    def __call__(self, x):
        """x: [batch, seq_len, d_model]"""
        mean = np.mean(x, axis=-1, keepdims=True)  # mean per token
        var  = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# Activation: GELU
def gelu(x):
    """Approximate GELU activation"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))

# Feed Forward Network
class FeedForward:
    """FFN dua lapis: Linear -> akt -> Linear
    - d_model: dim model
    - d_ff: dim hidden
    - activation: fungsi aktivasi (gelu atau np.tanh / np.relu)
    """
    def __init__(self, d_model, d_ff, activation='gelu', rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.W1 = rng.normal(scale=(d_model ** -0.5), size=(d_model, d_ff))
        self.b1 = np.zeros((d_ff,))
        self.W2 = rng.normal(scale=(d_ff ** -0.5), size=(d_ff, d_model))
        self.b2 = np.zeros((d_model,))
        if activation == 'gelu':
            self.act = gelu
        elif activation == 'relu':
            self.act = lambda x: np.maximum(0, x)
        elif activation == 'tanh':
            self.act = np.tanh
        else:
            raise ValueError("activation harus 'gelu', 'relu', atau 'tanh'")

    def __call__(self, x):
        # x: [batch, seq, d_model]
        y = x @ self.W1 + self.b1  # -> [batch, seq, d_ff]
        y = self.act(y)
        y = y @ self.W2 + self.b2  # -> [batch, seq, d_model]
        return y

# Scaled Dot-Product Attention
def ScaledDotProductAttention(q, k, v, mask=None):
    """Hitung attention:
    q,k,v: [batch, n_heads, seq, head_dim]
    mask: additive mask broadcastable ke [batch, n_heads, seq, seq]
    mengembalikan: context [batch, n_heads, seq, head_dim], attn_weights [batch, n_heads, seq, seq]
    """
    # skor raw: q @ k.T
    # q @ k.transpose(..., -1, -2)
    dk = q.shape[-1]
    scores = q @ k.transpose(0,1,3,2)  # -> [batch, n_heads, seq, seq]
    scores = scores / np.sqrt(dk)
    if mask is not None:
        scores = scores + mask  # mask additive (gunakan -1e9 untuk menekan)
    attn_weights = softmax_stable(scores, axis=-1)  # normalisasi
    context = attn_weights @ v  # -> [batch, n_heads, seq, head_dim]
    return context, attn_weights

# Multi-Head Attention
class MultiHeadAttention:
    """Multi-head attention lengkap dengan proyeksi Q,K,V dan output.
    - d_model: dimensi model
    - n_heads: jumlah head
    """
    def __init__(self, d_model, n_heads, rng=None):
        assert d_model % n_heads == 0, "d_model harus dibagi n_heads"
        if rng is None:
            rng = np.random.default_rng(0)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # bobot proyeksi
        self.W_q = rng.normal(scale=(d_model ** -0.5), size=(d_model, d_model))
        self.W_k = rng.normal(scale=(d_model ** -0.5), size=(d_model, d_model))
        self.W_v = rng.normal(scale=(d_model ** -0.5), size=(d_model, d_model))
        self.W_o = rng.normal(scale=(d_model ** -0.5), size=(d_model, d_model))

    def split_heads(self, x):
        # x: [batch, seq, d_model] -> [batch, n_heads, seq, head_dim]
        b, seq, _ = x.shape
        x = x.reshape(b, seq, self.n_heads, self.head_dim)
        return x.transpose(0,2,1,3)

    def combine_heads(self, x):
        # x: [batch, n_heads, seq, head_dim] -> [batch, seq, d_model]
        b, nh, seq, hd = x.shape
        x = x.transpose(0,2,1,3).reshape(b, seq, nh*hd)
        return x

    def __call__(self, x, mask=None):
        """x: [batch, seq, d_model]
        mask: additive mask shape (seq, seq) atau broadcastable
        return: out [batch, seq, d_model], attn_weights [batch, n_heads, seq, seq]
        """
        b, seq, _ = x.shape
        q = x @ self.W_q  # [b, seq, d_model]
        k = x @ self.W_k
        v = x @ self.W_v

        # split heads
        q = self.split_heads(q)  # [b, nh, seq, head_dim]
        k = self.split_heads(k)
        v = self.split_heads(v)

        # mask: jika bentuk (seq, seq) -> tambahkan dim untuk broadcast
        if mask is not None:
            # mask shape (seq, seq) -> broadcast ke (b, nh, seq, seq)
            mask_batched = mask[np.newaxis, np.newaxis, :, :]
        else:
            mask_batched = None

        context, attn_weights = ScaledDotProductAttention(q, k, v, mask=mask_batched)
        # gabungkan heads
        concat = self.combine_heads(context)  # [b, seq, d_model]
        out = concat @ self.W_o  # project output
        return out, attn_weights
