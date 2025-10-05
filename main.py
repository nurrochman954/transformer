import numpy as np
from transformer import DecoderOnlyModel, visualize_attention

def main():
    np.random.seed(42)

    # buat daftar token contoh (200 item)
    vocab_list = [f"TOKEN_{i}" for i in range(200)]

    # inisialisasi model dengan weight tying aktif
    model = DecoderOnlyModel(
        vocab_size=len(vocab_list),
        d_model=64,
        n_heads=4,
        d_ff=128,
        num_layers=2,
        max_len=64,
        pos_method='sinusoidal',
        weight_tying=True,
        id_to_token=vocab_list
    )

    # buat dua input random (batch=2, seq_len=12)
    tokens = np.random.randint(0, len(vocab_list), size=(2, 12))

    # jalankan forward pass dan tampilkan hasil penjelasan
    logits, probs_last, attn_maps = model.forward_verbose(tokens)

    # tampilkan heatmap attention (layer 0, head 0)
    visualize_attention(attn_maps, layer=0, head=0, sample=0)

if __name__ == "__main__":
    main()
