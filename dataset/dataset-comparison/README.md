# 📊 Dataset Comparator

**DatasetComparator** adalah alat analisis data Python yang digunakan untuk membandingkan dua dataset berdasarkan *data quality metrics* seperti **kelengkapan**, **keunikan**, **konsistensi**, dan **akurasi**. Proyek ini sangat berguna untuk data cleaning, pipeline validasi, dan analisis perbandingan antar versi dataset.

---

## 🚀 Fitur Utama

- 🔍 **Perbandingan kualitas data** berdasarkan 4 metrik utama.
- 🔗 **Analisis overlap** untuk mendeteksi duplikasi atau kesamaan antar dataset.
- 📊 **Visualisasi perbandingan** dalam bentuk grafik batang.
- 📝 **Laporan HTML interaktif** yang mendokumentasikan hasil perbandingan secara lengkap.
- 🧠 **Deteksi otomatis key column** untuk pencocokan data antar baris.

---

## 📁 Struktur Proyek

```
.
├── dataset/
│   ├── houses.csv
│   └── houses-cleaned.csv
├── dataset-comparison/
│   ├── comparison.py
│   ├── requirements.txt
│   └── reports
│       └── dataset_comparison_<timestamp>.html
```

---

## ⚙️ Cara Menjalankan

### 1. Instalasi Dependensi

Gunakan `pip` untuk menginstal dependensi yang dibutuhkan:

```bash
pip install -r requirements.txt
```

### 2. Jalankan Program

```bash
python comparison.py
```

Output:
- Perbandingan ditampilkan di terminal.
- Laporan `.html` dan grafik `.png` (sementara) akan disimpan di folder `./reports`.

---

## 📐 Metrik Penilaian

| Metrik        | Deskripsi                                                                 |
|---------------|---------------------------------------------------------------------------|
| Completeness  | Persentase data yang tidak hilang (non-null values).                     |
| Uniqueness    | Tingkat keunikan baris (bebas duplikasi).                                |
| Consistency   | Konsistensi data berdasarkan deteksi outlier atau format tidak sesuai.   |
| Accuracy      | Validasi nilai berdasarkan pola dan tipe data umum.                      |

---

## 📄 Contoh Output Laporan

Laporan HTML akan mencakup:
- Ringkasan pemenang (dataset yang lebih bersih)
- Tabel skor lengkap per metrik
- Grafik batang visualisasi perbandingan
- Catatan perbedaan signifikan antar dataset

---

## 📌 Catatan Teknis

- Jika tidak ditemukan kolom kunci yang eksplisit (`id`, `name`, `code`, dll), maka perbandingan akan menggunakan seluruh kolom yang sama.
- Untuk dataset besar (>5000 baris), analisis overlap dilakukan menggunakan **sampling** agar performa tetap optimal.

---

## 📬 Kontak

Dikembangkan oleh [Fajar Jati Nugroho](https://github.com/FjrREPO).  
Feel free to open an issue or pull request!

---

## 📜 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).