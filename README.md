# Skin Cancer Classification with SVM

Proyek machine learning untuk mengklasifikasikan citra lesi kulit ke dalam dua kelas, **benign** (jinak) dan **malignant** (ganas), menggunakan ekstraksi fitur citra dan algoritma Support Vector Machine (SVM).

> **Disclaimer:** Proyek ini dibuat untuk pembelajaran dan portofolio. Hasil model bukan diagnosis medis dan tidak boleh digunakan sebagai pengganti pemeriksaan tenaga medis profesional.

## Ringkasan Proyek

Notebook ini mengimplementasikan alur klasifikasi citra sebagai berikut:

1. **Preprocessing** — mengubah ukuran citra menjadi 128 × 128 piksel dan mengonversi warna ke RGB.
2. **Segmentasi lesi** — menggunakan K-Means clustering (`k=2`) dan mengasumsikan cluster dengan nilai warna rata-rata paling gelap sebagai area lesi.
3. **Post-processing** — menghapus objek kecil dari mask segmentasi menggunakan operasi morfologi.
4. **Ekstraksi fitur** — mengambil fitur tekstur GLCM (`contrast`, `correlation`, `energy`, `homogeneity`) serta fitur bentuk (`area`, `perimeter`).
5. **Normalisasi fitur** — menggunakan `StandardScaler`.
6. **Klasifikasi** — membandingkan SVM kernel `linear`, `poly`, dan `rbf`.
7. **Tuning dan evaluasi** — melakukan pencarian parameter dengan `GridSearchCV` serta mengevaluasi model menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.

## Hasil Eksperimen

Berdasarkan output yang tersimpan di notebook:

| Model | Accuracy |
|---|---:|
| SVM Linear | 79.5% |
| SVM Polynomial | 76.9% |
| SVM RBF | 81.2% |
| SVM RBF setelah tuning | **83.0%** |

Hasil tersebut berasal dari output notebook yang tersedia dan dapat berubah jika dataset, lingkungan, atau parameter eksperimen berbeda. Sebelum menggunakan angka ini dalam laporan, jalankan ulang notebook untuk memverifikasi hasil.

## Dataset

Dataset yang digunakan adalah **Melanoma Skin Cancer Dataset of 10,605 Images** dari Kaggle oleh Hasnain Javed.

- **Tautan dataset:** https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
- Silakan baca ketentuan penggunaan dan lisensi dataset di halaman sumber sebelum mendistribusikan data atau menggunakannya untuk tujuan lain.
- Dataset tidak disertakan dalam repository ini. Unduh langsung dari sumbernya.

Setelah mengunduh dan mengekstrak dataset, susun folder agar sesuai dengan struktur berikut (sesuaikan jika nama folder dari arsip Kaggle berbeda):

```text
data/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

## Struktur Repository

```text
skin-cancer-classification-portfolio/
├── notebooks/
│   └── skin_cancer_classification.ipynb
├── data/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── benign/
│       └── malignant/
├── results/
│   └── .gitkeep
├── assets/
│   └── .gitkeep
├── .gitignore
├── requirements.txt
└── README.md
```

Folder `data/` sengaja tidak berisi gambar dan diabaikan oleh Git agar repository tidak membengkak serta menghormati ketentuan distribusi dataset.

## Instalasi dan Cara Menjalankan

Disarankan menggunakan Python 3.10 atau 3.11.

1. Clone repository:

   ```bash
   git clone https://github.com/USERNAME/skin-cancer-classification-portfolio.git
   cd skin-cancer-classification-portfolio
   ```

2. Buat dan aktifkan virtual environment:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **Linux/macOS**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Pasang dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Unduh dataset dari Kaggle, ekstrak, dan letakkan ke folder `data/` sesuai struktur di atas.

5. Jalankan Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Buka `notebooks/skin_cancer_classification.ipynb` dan jalankan sel secara berurutan dari atas ke bawah.

Jika notebook tidak menemukan dataset, periksa kembali struktur folder dan pastikan nama kelas adalah `benign` serta `malignant`.

## Teknologi

- Python
- OpenCV
- NumPy dan Pandas
- Matplotlib dan Seaborn
- scikit-image
- scikit-learn
- Jupyter Notebook

## Catatan Pengembangan

- Validasi hasil segmentasi secara visual karena asumsi bahwa lesi selalu menjadi cluster paling gelap tidak selalu benar untuk semua citra.
- Pastikan preprocessing dan normalisasi diterapkan secara konsisten; untuk eksperimen yang lebih ketat, `StandardScaler` sebaiknya di-fit hanya pada data training lalu digunakan untuk mentransformasi data test.
- Evaluasi model dengan data yang benar-benar terpisah dan hindari kebocoran data saat tuning.
- Pertimbangkan metrik tambahan seperti ROC-AUC dan analisis false negative, terutama karena konteksnya adalah klasifikasi lesi kulit.

## Author

**Ryan Muhammad Hildan**

Proyek portofolio machine learning / computer vision.
