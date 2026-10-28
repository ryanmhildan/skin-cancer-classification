# Skin Cancer Classification with SVM

Klasifikasi citra lesi kulit menjadi **benign** (jinak) dan **malignant** (ganas) menggunakan segmentasi K-Means, fitur tekstur GLCM, dan Support Vector Machine.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-F7931E?logo=scikitlearn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-image%20processing-5C3EE8?logo=opencv&logoColor=white)

> **Disclaimer:** Proyek ini dibuat untuk pembelajaran dan portofolio. Hasil model bukan diagnosis medis dan tidak boleh digunakan sebagai pengganti pemeriksaan tenaga medis profesional.

## Daftar Isi

- [Ringkasan](#ringkasan)
- [Alur Kerja](#alur-kerja)
- [Tangkapan Layar](#tangkapan-layar)
- [Hasil Eksperimen](#hasil-eksperimen)
- [Dataset](#dataset)
- [Struktur Repository](#struktur-repository)
- [Instalasi dan Cara Menjalankan](#instalasi-dan-cara-menjalankan)
- [Teknologi](#teknologi)
- [Catatan Pengembangan](#catatan-pengembangan)
- [Author](#author)

## Ringkasan

| | |
|---|---|
| Tugas | Klasifikasi biner citra lesi kulit (benign vs malignant) |
| Pendekatan | Pengolahan citra klasik dan machine learning, tanpa deep learning |
| Fitur | 4 fitur tekstur GLCM dan 2 fitur bentuk |
| Model terbaik | SVM RBF (`C=100`, `gamma=1`) |
| Akurasi terbaik | **83,0%** pada 1.000 citra test |

## Alur Kerja

1. **Preprocessing:** citra diubah ukurannya menjadi 128 × 128 piksel dan dikonversi ke RGB.
2. **Segmentasi lesi:** K-Means clustering (`k=2`). Cluster dengan rata-rata warna paling gelap dianggap sebagai area lesi.
3. **Post-processing:** objek kecil pada mask dihapus dengan operasi morfologi.
4. **Ekstraksi fitur:** tekstur GLCM (`contrast`, `correlation`, `energy`, `homogeneity`) dan bentuk (`area`, `perimeter`).
5. **Normalisasi:** `StandardScaler` di-fit pada data training, lalu dipakai untuk mentransformasi data test.
6. **Klasifikasi:** perbandingan SVM kernel `linear`, `poly`, dan `rbf`.
7. **Tuning dan evaluasi:** `GridSearchCV` dengan 10-fold, lalu evaluasi memakai accuracy, precision, recall, F1-score, dan confusion matrix.

## Tangkapan Layar

### Hasil Segmentasi Lesi

Dari kiri ke kanan: citra asli, mask lesi hasil K-Means dan morfologi, serta area lesi yang tersegmentasi.

![Visualisasi segmentasi lesi](assets/segmentation_sample.png)

### Confusion Matrix (SVM RBF setelah tuning)

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Confusion matrix SVM RBF setelah tuning" width="480">
</p>

Dari 1.000 citra test, model memprediksi benar 420 citra benign dan 410 citra malignant. Ada 90 citra malignant yang terprediksi sebagai benign (false negative), dan kesalahan jenis ini paling perlu diperhatikan dalam konteks medis.

## Hasil Eksperimen

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| SVM Linear | 79,5% | 0,80 | 0,79 | 0,79 |
| SVM Polynomial | 76,9% | 0,80 | 0,77 | 0,76 |
| SVM RBF | 81,2% | 0,82 | 0,81 | 0,81 |
| SVM RBF + tuning (`C=100`, `gamma=1`) | **83,0%** | **0,83** | **0,83** | **0,83** |

Rincian per kelas untuk SVM RBF setelah tuning:

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| benign | 0,82 | 0,84 | 0,83 | 500 |
| malignant | 0,84 | 0,82 | 0,83 | 500 |

Angka di atas diambil dari output yang tersimpan di notebook dan bisa berubah jika dataset, lingkungan, atau parameter berbeda.

## Dataset

[Melanoma Skin Cancer Dataset of 10,605 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) dari Kaggle oleh Hasnain Javed.

Dataset tidak disertakan dalam repository ini, jadi unduh langsung dari sumbernya. Baca juga ketentuan penggunaan dan lisensinya sebelum mendistribusikan atau memakainya untuk tujuan lain.

Setelah diunduh dan diekstrak, susun folder seperti berikut (sesuaikan jika nama folder dari arsip Kaggle berbeda):

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
skin-cancer-classification/
├── assets/                # gambar untuk README
│   ├── segmentation_sample.png
│   └── confusion_matrix.png
├── data/                  # dataset (diabaikan Git, unduh sendiri)
├── notebooks/
│   └── skin_cancer_classification.ipynb
├── results/               # output eksperimen
├── .gitignore
├── requirements.txt
└── README.md
```

Folder `data/` sengaja diabaikan oleh Git agar repository tetap ringan dan sesuai dengan ketentuan distribusi dataset.

## Instalasi dan Cara Menjalankan

Disarankan menggunakan Python 3.10 atau 3.11.

1. Clone repository:

   ```bash
   git clone https://github.com/ryanmhildan/skin-cancer-classification.git
   cd skin-cancer-classification
   ```

2. Buat dan aktifkan virtual environment:

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Pasang dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Unduh dataset dari Kaggle, ekstrak, lalu letakkan di folder `data/` sesuai struktur di atas.

5. Jalankan Jupyter Notebook, buka `notebooks/skin_cancer_classification.ipynb`, dan jalankan sel secara berurutan:

   ```bash
   jupyter notebook
   ```

Notebook ini awalnya dibuat di Google Colab dan membaca dataset dari Google Drive. Untuk menjalankannya secara lokal, lewati sel `drive.mount(...)` dan ubah `DATASET_TRAIN_PATH` serta `DATASET_TEST_PATH` menjadi `data/train` dan `data/test`. Pastikan nama folder kelas adalah `benign` dan `malignant`.

## Teknologi

- Python
- OpenCV dan scikit-image
- NumPy dan Pandas
- scikit-learn
- Matplotlib dan Seaborn
- Jupyter Notebook / Google Colab

## Catatan Pengembangan

- Hasil segmentasi perlu dicek secara visual. Asumsi bahwa lesi selalu menjadi cluster paling gelap tidak selalu benar, misalnya pada citra dengan tepi gelap seperti contoh di atas.
- `StandardScaler` dan tuning hanya boleh memakai data training agar tidak terjadi kebocoran data.
- Metrik seperti ROC-AUC dan analisis false negative layak ditambahkan karena konteksnya adalah deteksi lesi ganas.
- Untuk pengembangan berikutnya, bisa dicoba fitur warna (histogram, color moments), fitur bentuk tambahan (asimetri, kebulatan), atau CNN sebagai pembanding.

## Author

**Ryan Muhammad Hildan** ([GitHub](https://github.com/ryanmhildan))

Proyek portofolio machine learning dan computer vision.
