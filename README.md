# Transformer Decoder-Only (GPT-Style) from Scratch

Implementasi arsitektur **Transformer Decoder-Only (GPT-style)** dari nol menggunakan **NumPy**, tanpa library deep learning seperti PyTorch atau TensorFlow.  
Proyek ini dibuat untuk memahami cara kerja internal Transformer, mulai dari *token embedding*, *positional encoding*, *multi-head self-attention*, hingga *causal masking* dan prediksi token berikutnya.

## Latar Belakang Singkat

Model ini terinspirasi dari arsitektur **GPT (Generative Pre-trained Transformer)**, yaitu varian *decoder-only Transformer* yang bekerja secara **autoregressive** dengan menghasilkan token berikutnya berdasarkan token-token sebelumnya.  
Tujuan utama proyek ini untuk **membangun dan memahami setiap komponen Transformer secara matematis dan konseptual**.

Fitur tambahan yang disertakan dalam proyek ini:
- **Weight Tying**: Bobot embedding input dan output menggunakan matriks yang sama untuk efisiensi parameter.
- **Visualisasi Attention**: Menampilkan *attention heatmap* agar hubungan antar token dapat diamati secara visual.

## Konfigurasi & Menjalankan

### 1. Persiapan Awal

Pastikan Anda memiliki **Python 3.9** atau yang lebih baru.

- Clone repositori ini
   ```
   git clone https://github.com/nurrochman954/transformer.git
   cd transformer
   ```
   
- Buat dan aktifkan virtual environment (disarankan)  
    ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

- Install dependensi  
    ```bash
    pip install -r requirements.txt
    ```


### 2. Menjalankan Program

Jalankan file utama untuk melakukan pengujian sederhana

    ```bash
    python main.py
    ```

### 4. Pengujian

Hasil pengujian memastikan bahwa semua komponen bekerja dengan benar:
- Bentuk tensor sesuai spesifikasi `[batch, seq_len, vocab_size]`.  
- Distribusi softmax valid, dengan jumlah probabilitas per batch = 1.  
- Causal mask berfungsi dengan benar — *future tokens* tidak terlihat pada area di atas diagonal.  
- Model mampu memprediksi token berikutnya dengan benar (*Top-5 tokens* dengan probabilitas tertinggi).


**Catatan Tambahan**: 
- Model ini **tidak melibatkan proses training**, hanya *forward pass*
- Seluruh komputasi arsitektur dilakukan **murni menggunakan NumPy**, termasuk embedding, attention, dan feed-forward network.
- Library **Matplotlib** digunakan **secara terbatas untuk visualisasi hasil** (*attention heatmap*).
- Visualisasi ini membantu memahami bagaimana model memberikan *attention* pada token sebelumnya saat memprediksi token baru.

## Visualisasi Attention Heatmap
Gambar berikut menunjukkan hasil visualisasi *attention heatmap* dari model GPT-style yang dibangun.  
Area berwarna terang menunjukkan *active attention* antar token (past dan current positions),csementara area gelap di atas diagonal menunjukkan *masked future tokens*.

![Attention Heatmap](https://github.com/nurrochman954/transformer/blob/main/Visualisasi%20attention%20heatmap.png)
