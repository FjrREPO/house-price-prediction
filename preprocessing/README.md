# 🧹 Preprocessing Data

Repositori ini berisi script Python untuk **membersihkan dan memproses data mentah properti** (real estate) dari wilayah Yogyakarta. Data mentah berasal dari hasil scraping dan disimpan dalam format CSV. Proses preprocessing mencakup pembersihan data, transformasi nilai, serta validasi kelengkapan dan konsistensi data.

## 📌 Tujuan

Menyiapkan data real estate yang bersih dan terstruktur sebagai input untuk model prediksi harga rumah berbasis logika fuzzy atau machine learning.

## 📂 Struktur Folder

```
.
├── dataset/
│   ├── houses.csv              # Dataset mentah dari hasil scraping
│   └── houses-cleaned.csv      # Output setelah preprocessing
├── preprocessing.py            # Script utama untuk preprocessing
└── requirements.txt            # Dependencies yang dibutuhkan
```

## ⚙️ Proses Preprocessing

Berikut adalah langkah-langkah utama dalam preprocessing:

1. **Menghapus duplikasi data**
2. **Membuang data dengan nilai kosong atau tidak valid**
3. **Mengonversi kolom harga ke format numerik (IDR)**
4. **Parsing lokasi ke dalam format kota/kabupaten**
5. **Konversi tipe data ke bentuk yang sesuai**
6. **Penghapusan outlier (opsional)**
7. **Validasi akhir dan ekspor ke file `houses-cleaned.csv`**

## 📦 Library yang Digunakan

- `pandas`
- `numpy`
- `re` (regex untuk parsing teks)
- `sklearn.preprocessing` (jika diperlukan normalisasi)

## ▶️ Cara Menjalankan

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Jalankan script:

```bash
python preprocessing.py
```

Dataset hasil preprocessing akan tersedia di `dataset/houses-cleaned.csv`.

## 📌 Catatan

- Script ini **belum mencakup prediksi harga** — hanya pembersihan dan persiapan data.
- Format harga dikonversi dari satuan seperti "Rp 300 jt" atau "Rp 1,2 M" ke format numerik seperti `300000000` atau `1200000000`.

## 👨‍💻 Kontribusi

Silakan buka issue jika ada bug atau request fitur baru. Pull request sangat disambut!