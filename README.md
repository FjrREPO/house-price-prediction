# Implementasi Fuzzy Logic dan Optimasi Algoritma Genetika pada Random Forest untuk Prediksi Harga Rumah di Kota Yogyakarta

## Deskripsi

Penentuan harga rumah merupakan tantangan kompleks yang sering dihadapi oleh industri properti, terutama ketika menangani data yang tidak pasti atau ambigu seperti lokasi dan kualitas bangunan. Metode konvensional seperti regresi linier sering kali gagal memberikan prediksi yang akurat karena tidak mampu menangani kompleksitas data secara efektif.

Penelitian ini mengimplementasikan berbagai teknik pembelajaran mesin untuk memprediksi harga rumah, termasuk model **Fuzzy Logic**, **Random Forest**, dan **Hyperparameter Tuning** untuk meningkatkan akurasi. Selain itu, proyek ini juga mencakup **data preprocessing**, **analisis fitur**, serta **evaluasi model** menggunakan beberapa metrik seperti MAE, RMSE, MAPE, dan R².

### Metode yang Digunakan:
- **Fuzzy Logic** untuk menangani data dengan ketidakpastian dan memberikan keputusan yang lebih fleksibel.
- **Random Forest Regressor** untuk model prediksi berbasis ensemble.
- **Hyperparameter Tuning** menggunakan Grid Search untuk mencari kombinasi parameter terbaik.
- **Cross-validation** untuk mengevaluasi model secara lebih mendalam.

## Struktur Proyek

Berikut adalah struktur proyek yang digunakan dalam penelitian ini:

```
.
├── Home.py                          # Skrip utama untuk menjalankan aplikasi Streamlit
├── requirements.txt                 # Daftar dependensi Python untuk proyek ini
├── dataset                          # Folder dataset
│   ├── houses-cleaned.csv           # Data rumah yang telah dibersihkan
│   ├── dataset-comparison           # Folder untuk perbandingan dataset
│   └── dataset-scraper              # Skrip dan data untuk scraping real estate
├── model                            # Folder model yang berisi model yang telah dilatih dan evaluasi
│   ├── model.joblib                 # Model Random Forest yang telah dilatih
│   └── evaluation                   # Folder untuk hasil evaluasi model
│       ├── actual_vs_predicted.png  # Visualisasi perbandingan harga aktual vs prediksi
│       ├── feature_importance.png   # Visualisasi pentingnya fitur
│       ├── residuals.png            # Visualisasi residuals model
│       └── prediction_info.json     # Informasi tentang model dan fitur
├── preprocessing                    # Folder untuk preprocessing data
│   ├── preprocessing.py            # Skrip untuk preprocessing data
│   ├── output                       # Hasil preprocessing
│   └── evaluation                   # Hasil evaluasi preprocessing
├── pages                            # Folder untuk aplikasi Streamlit
│   ├── 1_Dataset.py                # Halaman untuk memuat dan menampilkan dataset
│   ├── 2_Preprocessing.py          # Halaman untuk preprocessing data
│   └── 3_Model.py                  # Halaman untuk melatih dan mengevaluasi model
└── .gitignore                      # File untuk mengecualikan file tertentu dari git
```

### Skrip dan Penjelasan:
1. **Home.py** - Skrip utama untuk menjalankan aplikasi Streamlit. Di sini, pengguna dapat mengakses seluruh pipeline dan visualisasi interaktif dari dataset hingga evaluasi model.
2. **preprocessing.py** - Skrip yang bertanggung jawab untuk membersihkan dan mempersiapkan data. Data rumah dibersihkan, dihapuskan yang tidak sesuai, dan diproses untuk model.
3. **Model Evaluation** - Melakukan evaluasi terhadap model yang dilatih, termasuk perbandingan model sebelum dan sesudah hyperparameter tuning, serta visualisasi dari hasil evaluasi (misalnya, **Actual vs Predicted**).

## Persiapan Lingkungan

### Instalasi Dependensi:
1. **Clone repository ini**:
   ```bash
   git clone https://github.com/FjrREPO/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Aktifkan environment virtual**:
   ```bash
   source venv/bin/activate
   ```

4. **Menjalankan aplikasi Streamlit**:
   ```bash
   streamlit run Home.py
   ```

### Dependencies:
- `scikit-learn` (untuk model prediksi dan evaluasi)
- `pandas` (untuk manipulasi data)
- `numpy` (untuk komputasi numerik)
- `matplotlib` dan `seaborn` (untuk visualisasi)
- `joblib` (untuk menyimpan dan memuat model)
- `streamlit` (untuk antarmuka pengguna interaktif)
- `colorama` dan `tabulate` (untuk tampilan hasil yang lebih informatif di terminal)

## Proses Prediksi Harga

### Pipeline:
1. **Data Loading** - Data dimuat dari file CSV dan dibersihkan.
2. **Model Training** - Model Random Forest dilatih dengan data pelatihan.
3. **Cross-Validation** - Model divalidasi dengan menggunakan cross-validation untuk mengevaluasi performa.
4. **Feature Importance** - Menilai fitur-fitur yang paling berpengaruh terhadap model.
5. **Hyperparameter Tuning** - Mencari kombinasi terbaik dari hyperparameters model.
6. **Model Evaluation** - Mengukur performa model menggunakan MAE, RMSE, MAPE, dan R².
7. **Model Saving** - Model yang telah dilatih dan disempurnakan disimpan dalam file `.joblib` untuk digunakan lebih lanjut.

### Menjalankan Prediksi:
Jika Anda ingin menggunakan model yang telah dilatih untuk melakukan prediksi harga rumah, Anda dapat menggunakan fungsi `predict` yang tersedia di kelas `HousePriceModel`.

```python
features_dict = {
    "city": "Yogyakarta",
    "bedrooms": 3,
    "bathrooms": 2,
    "lt": 150,
    "lb": 100,
}

model = HousePriceModel(data_path="dataset/houses-cleaned.csv")
predicted_price = model.predict(features_dict)
print(f"Predicted price: Rp {predicted_price:,.0f}")
```

---

## 📬 Kontak

Dikembangkan oleh [Fajar Jati Nugroho](https://github.com/FjrREPO).  
Feel free to open an issue or pull request!

---

## 📜 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
```

This updated `README.md` includes the full project structure, usage instructions, and details about the methodology, helping others to understand how to run and contribute to the project.