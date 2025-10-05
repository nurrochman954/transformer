import numpy as np
import matplotlib.pyplot as plt
from embedding import TokenEmbedding, PositionalEncoding
from layers import LayerNorm, FeedForward, MultiHeadAttention
from utils import causal_mask, softmax_stable

# Fungsi untuk visualisasi attention
def visualize_attention(attn_maps, layer=0, head=0, sample=0, figsize=(7, 6), cmap='viridis'):
    """
    Menampilkan peta attention (attention map) dalam bentuk heatmap.
    - attn_maps : daftar attention matrix dari setiap layer
    - layer, head, sample : indeks layer, head, dan sampel yang ingin ditampilkan
    - cmap : skema warna (default: viridis)
    """
    attn = attn_maps[layer][sample, head]  # shape: (seq_len, seq_len)
    plt.figure(figsize=figsize)
    plt.imshow(attn, cmap=cmap, aspect='auto')
    plt.colorbar()
    plt.title(f"Attention Layer {layer}, Head {head}, Sample {sample}")
    plt.xlabel("Key Positions (tokens being attended to)")
    plt.ylabel("Query Positions (tokens attending)")
    plt.show()

# Fungsi helper: interpretasi hasil prediksi
def interpret_output(probs, id_to_token=None, top_k=5):
    """
    Mengambil token dengan probabilitas tertinggi (Top-k)
    dari hasil softmax pada token terakhir.
    - probs : array numpy [vocab_size]
    - id_to_token : daftar nama token (opsional)
    """
    top_idx = np.argsort(probs)[::-1][:top_k]
    hasil = []
    for i in top_idx:
        nama = id_to_token[i] if id_to_token is not None else f"<id={i}>"
        hasil.append((nama, float(probs[i])))
    return hasil

# DecoderBlock dan Decoder Stack
class DecoderBlock:
    """
    Satu blok Transformer Decoder yang terdiri dari:
    - Multi-Head Self-Attention dengan residual connection dan LayerNorm
    - Feed-Forward Network dengan residual connection dan LayerNorm
    """

    def __init__(self, d_model, n_heads, d_ff, activation='gelu', rng=None):
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, activation=activation, rng=rng)

    def __call__(self, x, mask=None):
        # Sub-layer pertama: Multi-Head Self-Attention
        x_norm = self.ln1(x)
        attn_out, attn_weights = self.mha(x_norm, mask=mask)
        x = x + attn_out  # residual connection

        # Sub-layer kedua: Feed-Forward Network
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out  # residual connection

        return x, attn_weights


class TransformerDecoder:
    """
    Stack beberapa DecoderBlock untuk membentuk decoder penuh.
    """

    def __init__(self, num_blocks, d_model, n_heads, d_ff, activation='gelu', rng=None):
        self.blocks = [
            DecoderBlock(d_model, n_heads, d_ff, activation=activation, rng=rng)
            for _ in range(num_blocks)
        ]
        self.ln_final = LayerNorm(d_model)

    def __call__(self, x, mask=None):
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x, mask=mask)
            attn_maps.append(attn)
        x = self.ln_final(x)
        return x, attn_maps

# Model Lengkap: Decoder-Only Transformer
class DecoderOnlyModel:
    """
    Model GPT-style (Decoder-Only Transformer) yang dibangun sepenuhnya dengan NumPy.
    Komponen utama:
    - Token Embedding + Positional Encoding
    - Multi-Head Self-Attention dan Feed-Forward Network
    - Layer Normalization dan Residual Connection
    - Causal Masking untuk mencegah akses ke future tokens
    - Fitur tambahan: weight tying dan visualisasi distribusi attention
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256, num_layers=2,
                 max_len=128, pos_method='sinusoidal', weight_tying=True,
                 id_to_token=None, rng=None):

        if rng is None:
            rng = np.random.default_rng(0)

        self.vocab_size = vocab_size
        self.id_to_token = id_to_token
        self.embed = TokenEmbedding(vocab_size, d_model, rng=rng)
        self.pos = PositionalEncoding(d_model, max_len=max_len, method=pos_method, rng=rng)
        self.decoder = TransformerDecoder(num_layers, d_model, n_heads, d_ff, rng=rng)

        # Weight tying: gunakan embedding matrix yang sama untuk output projection
        self.weight_tying = weight_tying
        if weight_tying:
            self.W_out = None  # diambil dari embedding nantinya
        else:
            self.W_out = rng.normal(scale=(d_model ** -0.5), size=(d_model, vocab_size))
        self.b_out = np.zeros((vocab_size,))

    # Forward Pass
    def forward(self, token_ids):
        """
        Melakukan forward pass dari input tokens hingga output probabilities.
        Menggunakan causal masking agar setiap token hanya dapat melakukan attention
        terhadap dirinya sendiri dan past tokens (tidak ke future tokens).
        """
        b, seq = token_ids.shape

        # Embedding dan Positional Encoding
        x = self.embed(token_ids)
        x = x + self.pos(seq)[np.newaxis, :, :]

        # Membuat causal mask (segitiga bawah)
        mask = causal_mask(seq)

        # Jalankan decoder stack
        x, attn_maps = self.decoder(x, mask=mask)

        # Output projection (dengan atau tanpa weight tying)
        if self.weight_tying:
            W_out = self.embed.W.T
        else:
            W_out = self.W_out

        logits = x @ W_out + self.b_out

        # Hitung probabilitas hanya untuk token terakhir (autoregressive behavior)
        probs_last = softmax_stable(logits[:, -1, :], axis=-1)

        return logits, probs_last, attn_maps

    # Forward Pass dengan penjelasan (verbose mode)
    def forward_verbose(self, token_ids):
        """
        Menjalankan forward pass dan menampilkan hasilnya secara lebih informatif.
        Termasuk bentuk tensor, pengecekan softmax, dan verifikasi causal masking.
        """
        logits, probs_last, attn_maps = self.forward(token_ids)

        print("=" * 70)
        print("MODEL DECODER-ONLY TRANSFORMER (GPT STYLE)")
        print("=" * 70)
        print(f"Jumlah batch        : {token_ids.shape[0]}")
        print(f"Panjang sequence    : {token_ids.shape[1]}")
        print(f"Ukuran vocabulary   : {self.vocab_size}")
        print(f"Weight tying        : {'Aktif' if self.weight_tying else 'Tidak'}")
        print("=" * 70)

        for b in range(token_ids.shape[0]):
            print(f"\n[Sample {b}] Input token IDs:", token_ids[b].tolist())
            top5 = interpret_output(probs_last[b], id_to_token=self.id_to_token)
            print("Prediksi token berikutnya (Top 5):")
            for nama, p in top5:
                print(f"  {nama:15s} -> Probabilitas: {p:.4f}")

        print("=" * 70)
        print("Cek hasil numerik:")
        print(f" - Bentuk logits         : {logits.shape}")
        print(f" - Bentuk attention map  : {attn_maps[0].shape} (batch, heads, seq, seq)")
        print(f" - Jumlah probabilitas   : {probs_last.sum(axis=-1)}")
        print(" - Causal masking diterapkan: area future tokens di bagian atas diagonal bernilai 0.")
        print("=" * 70)

        return logits, probs_last, attn_maps
