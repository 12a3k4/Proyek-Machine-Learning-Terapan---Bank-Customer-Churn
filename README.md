# 📊 Proyek Machine Learning Terapan - Bank Customer Churn
Submission Pertama dari course Machine Learning Terapan (Dicoding).

Dibuat dengan tujuan untuk melengkapi Learning Objective dari "Laskar AI 2025".

---

## Domain Permasalahan

Bank XYZ mengalami tingkat churn pelanggan yang cukup tinggi dalam beberapa tahun terakhir. Proyek ini bertujuan untuk memprediksi apakah seorang pelanggan akan berhenti (churn) berdasarkan data historis mereka. Dengan pemodelan yang tepat, pihak bank dapat menargetkan pelanggan berisiko tinggi dan melakukan upaya retensi secara efektif.

---

## Business Understanding

### Problem Statement
Berdasarkan data historis pelanggan, dapatkah kita membangun sebuah model machine learning yang mampu memprediksi apakah seorang pelanggan akan keluar (churn) atau tidak?

### Goals
- **Identifikasi Faktor Risiko:** Menentukan variabel-variabel utama (misalnya, kebiasaan makan, aktivitas fisik, dan riwayat keluarga) yang secara signifikan mempengaruhi tingkat obesitas.  
- **Pembangunan Model Prediktif:** Mengembangkan model machine learning yang dapat mengklasifikasikan tingkat obesitas dengan akurasi tinggi, sehingga memungkinkan identifikasi dini pada individu berisiko.  
- **Dukungan Keputusan Bisnis:** Menghasilkan output yang terukur dan dapat diintegrasikan ke dalam sistem pendukung keputusan, guna membantu lembaga kesehatan dan perusahaan asuransi dalam merancang program pencegahan dan intervensi yang efektif.

### Solution Statement (Opsional)
Solusi yang diusulkan melibatkan pengembangan pipeline data end-to-end, mulai dari pembersihan data dan transformasi fitur hingga penerapan beberapa algoritma machine learning untuk mengklasifikasikan apakah seorang pelanggan akan keluar (churn) atau tidak. Hasil prediksi akan dievaluasi dengan menggunakan metrik evaluasi yang komprehensif (akurasi, precision, recall, F1-score, dan confusion matrix). Model terbaik akan diintegrasikan ke dalam sistem pendukung keputusan untuk membantu perusahaan dalam mengoptimalkan retensi pelanggan, mengurangi kehilangan pendapatan, dan meningkatkan loyalitas pelanggan.

---
## Data Understanding

Dataset yang digunakan adalah Churn for Bank Customers yang terdiri dari 10000 baris dan 14 kolom. Data ini diperoleh dari https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers

### Ringkasan Data
- **Jumlah Data:**  
  Dataset ini mencakup 10000 baris (record) dan 14 kolom (fitur).

- **Kondisi Data:**  
  - **Missing Value:**  
    Pemeriksaan awal menunjukkan bahwa dataset tidak mengandung missing value, sehingga semua kolom memiliki entri yang lengkap.
  - **Duplikat:**  
    Ditemukan adanya baris duplikat yang kemudian dihapus untuk memastikan keunikan setiap record.
  - **Outlier:**  
    Analisis visual menggunakan boxplot mengidentifikasi adanya outlier pada beberapa variabel numerik, seperti pada fitur Height dan Weight, yang perlu dipertimbangkan dalam proses preprocessing.

### Uraian Fitur pada Data (direct translation dari https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)
1. RowNumber - nomor baris pada data.
2. CustomerId - nomor id untik setiap customer.
3. Surname - nama belakang setiap customer
4. CreditScore — dapat memengaruhi kemungkinan churn pelanggan, karena pelanggan dengan skor kredit yang lebih tinggi cenderung lebih kecil kemungkinannya untuk meninggalkan bank.
5. Geography — lokasi pelanggan dapat memengaruhi keputusan mereka untuk keluar dari bank.
6. Gender — menarik untuk diteliti apakah jenis kelamin memiliki peran dalam keputusan pelanggan untuk keluar dari bank.
7. Age — ini tentu saja relevan, karena pelanggan yang lebih tua cenderung lebih kecil kemungkinannya untuk keluar dari bank dibandingkan yang lebih muda.
8. Tenure — mengacu pada jumlah tahun pelanggan telah menjadi nasabah bank. Umumnya, pelanggan lama lebih loyal dan lebih kecil kemungkinannya untuk keluar dari bank.
9. Balance — merupakan indikator yang sangat baik terhadap churn pelanggan, karena pelanggan dengan saldo tinggi di rekening mereka cenderung lebih loyal dibandingkan mereka yang memiliki saldo rendah.
10. NumOfProducts — mengacu pada jumlah produk yang telah dibeli pelanggan melalui bank.
11. HasCrCard — menunjukkan apakah pelanggan memiliki kartu kredit. Kolom ini juga relevan, karena orang yang memiliki kartu kredit cenderung lebih kecil kemungkinan untuk keluar dari bank.
12. IsActiveMember — pelanggan yang aktif cenderung lebih kecil kemungkinan untuk meninggalkan bank.
13. EstimatedSalary — seperti halnya dengan saldo, pelanggan dengan gaji yang lebih rendah cenderung lebih mungkin untuk meninggalkan bank dibandingkan mereka yang bergaji lebih tinggi.
14. Exited - menentukan apakah customer tersebut sudah pergi meninggalkan bank atau belum.

Penjelasan identifikasi ini memberikan dasar untuk EDA


---

## Data Preparation

Proses Data Preparation telah dilakukan secara sistematis untuk memastikan data dalam kondisi optimal sebelum digunakan untuk pemodelan. Langkah-langkah yang telah dilakukan adalah sebagai berikut:

1. **Pembersihan Data:**  
    - Tidak ada missing value.
    - Duplikat dihapus.

2. **Encoding Variabel Kategorikal:**  
   - Kolom Gender dan Geography diubah menjadi numerik menggunakan LabelEncoder.

3. **Pemisahan Fitur dan Target:**  
   - Setelah proses encoding, fitur dan target telah dipisahkan dengan tepat.  
   - Target diambil dari kolom Exited, sedangkan seluruh kolom lainnya digunakan sebagai fitur untuk memprediksi tingkat obesitas.

4. **Train-Test Split:**  
   - Data dibagi menjadi data pelatihan dan data pengujian menggunakan metode train-test split.  
   - Pembagian dilakukan sebelum proses standarisasi untuk menghindari kebocoran informasi antara data pelatihan dan pengujian.

5. **Standarisasi Fitur Numerik:**  
   - Proses standarisasi dilakukan pada data pelatihan dengan StandardScaler, dan transformasi yang sama kemudian diterapkan pada data pengujian.  
   - Langkah ini memastikan bahwa seluruh fitur numerik berada pada skala yang seragam, yang penting untuk algoritma machine learning yang sensitif terhadap perbedaan skala.

---

## Model Development

Pada tahap pengembangan model, berbagai algoritma klasifikasi diterapkan untuk memprediksi apakah seorang pelanggan akan berhenti (churn). Berikut adalah penjelasan mengenai cara kerja masing-masing algoritma beserta parameter yang digunakan:

### Random Forest Classifier
Random Forest merupakan algoritma ensemble yang membangun banyak pohon keputusan (decision trees) dan menggabungkan hasil prediksi dari masing-masing pohon untuk menghasilkan keputusan akhir. Algoritma ini efektif untuk mengatasi overfitting dan menangani dataset dengan banyak fitur.  
- **Parameter Utama:**  
  - *n_estimators:* Jumlah pohon yang dibangun (default = 100).  
  - *max_depth:* Kedalaman maksimum pohon (default = None, yaitu tidak ada batasan).  
  - *max_features:* Jumlah fitur yang dipertimbangkan pada setiap split (default = "auto").

### Decision Tree Classifier
Decision Tree menggunakan struktur pohon untuk mengambil keputusan berdasarkan pembagian fitur secara berurutan. Model ini mudah diinterpretasikan, namun rawan overfitting jika tidak diatur dengan baik.  
- **Parameter Utama:**  
  - *criterion:* Metode untuk mengukur kualitas split (default = "gini", alternatif "entropy").  
  - *max_depth:* Kedalaman maksimum pohon (default = None).  
  - Parameter lain digunakan sesuai nilai default.

### AdaBoost Classifier
AdaBoost (Adaptive Boosting) merupakan metode ensemble yang menggabungkan beberapa model lemah untuk membentuk model yang kuat. Setiap model lemah diperbaiki berdasarkan kesalahan model sebelumnya.  
- **Parameter Utama:**  
  - *n_estimators:* Jumlah model lemah yang digabungkan (default = 50).  
  - *learning_rate:* Mengontrol kontribusi masing-masing model (default = 1.0).  
  - Parameter lain mengikuti nilai default.

### K-Nearest Neighbors (KNN)
KNN adalah algoritma non-parametrik yang mengklasifikasikan instance baru berdasarkan kedekatan (jarak) dengan data training. Model ini bergantung pada pemilihan jumlah tetangga terdekat (neighbors).  
- **Parameter Utama:**  
  - *n_neighbors:* Jumlah tetangga yang dipertimbangkan untuk prediksi (default = 5).  
  - Parameter lainnya menggunakan nilai default.

### Gradient Boosting Classifier
Gradient Boosting merupakan metode ensemble yang membangun model secara iteratif untuk mengurangi error residual dari model sebelumnya. Metode ini cenderung memberikan performa tinggi dan stabil.  
- **Parameter Utama:**  
  - *n_estimators:* Jumlah iterasi (model) yang dibangun (default = 100).  
  - *learning_rate:* Laju pembelajaran untuk setiap iterasi (default = 0.1).  
  - *max_depth:* Kedalaman maksimum pohon (default = 3).  
  - Parameter lain mengikuti nilai default.

### Logistic Regression
Logistic Regression adalah model klasifikasi linier yang menggunakan fungsi sigmoid untuk mengestimasi probabilitas suatu kelas. Model ini sederhana dan efektif terutama jika hubungan antara fitur dan target bersifat linier.  
- **Parameter Utama:**  
  - *max_iter:* Jumlah iterasi maksimum yang diijinkan untuk konvergensi (diatur ke 1000 untuk memastikan konvergensi).  
  - Parameter lainnya menggunakan nilai default.

Setiap model dilatih menggunakan data pelatihan yang telah dipreproses, dan performanya dievaluasi dengan menggunakan metrik seperti akurasi, precision, recall, F1 score, dan confusion matrix. Pemilihan model terbaik didasarkan pada hasil evaluasi yang komprehensif untuk memastikan model yang paling sesuai dengan kebutuhan prediksi tingkat obesitas.

# Evaluation

Bagian evaluasi menyajikan hasil metrik evaluasi dari setiap skema pelatihan dan melakukan komparasi untuk menentukan model terbaik. Hasil evaluasi ini kemudian dihubungkan dengan aspek bisnis dan tujuan proyek untuk memastikan bahwa solusi yang dikembangkan menjawab setiap problem statement dan mencapai goals yang diharapkan.

## Hasil Evaluasi Model

Beberapa model telah dilatih menggunakan data pelatihan yang telah dipreproses, dan evaluasi dilakukan pada data pengujian. Hasil evaluasi diperoleh berdasarkan metrik utama, yaitu:

- **Accuracy:** Persentase prediksi yang benar secara keseluruhan.
- **Precision:** Ketepatan prediksi positif.
- **Recall:** Kemampuan model mendeteksi semua instance positif.
- **F1 Score:** Rata-rata harmonis antara precision dan recall.
- **Confusion Matrix:** Distribusi prediksi terhadap label asli untuk masing-masing kelas.

### Hasil Metrik Evaluasi

Berikut adalah hasil evaluasi dari model klasifikasi yang diuji:

| Model                    | Accuracy | Precision | Recall  | F1 Score | Confusion Matrix           |
|--------------------------|----------|-----------|---------|----------|-----------------------------|
| Random Forest Classifier | 0.8664   | 0.8590    | 0.8664  | 0.8513   | [[1944, 59], [275, 222]]    |
| Decision Tree Classifier | 0.7820   | 0.7875    | 0.7820  | 0.7846   | [[1714, 289], [256, 241]]   |
| AdaBoost Classifier      | 0.8576   | 0.8465    | 0.8576  | 0.8446   | [[1917, 86], [270, 227]]    |
| KNN                      | 0.8252   | 0.8027    | 0.8252  | 0.8014   | [[1908, 95], [342, 155]]    |
| Gradient Boosting Classifier | 0.8684 | 0.8611    | 0.8684  | 0.8541   | [[1943, 60], [269, 228]]    |
| Logistic Regression      | 0.8128   | 0.7809    | 0.8128  | 0.7699   | [[1943, 60], [408, 89]]     |

**Model terbaik berdasarkan akurasi: Gradient Boosting Classifier**

### Komparasi dan Pemilihan Model Terbaik

Berdasarkan hasil evaluasi di atas, Gradient Boosting Classifier menempati posisi teratas dengan akurasi sebesar 86.84%, diikuti sangat dekat oleh Random Forest Classifier dengan akurasi 86.64%. Kedua model ini merupakan metode ensemble yang sangat efektif dalam menangani permasalahan prediksi churn pelanggan karena mampu menggabungkan kekuatan dari banyak model sederhana menjadi prediksi yang lebih akurat dan stabil.

AdaBoost Classifier juga menunjukkan performa yang kompetitif dengan akurasi 85.76%, menjadikannya alternatif yang layak jika model utama gagal diimplementasikan dalam sistem produksi.

Sementara itu, K-Nearest Neighbors (KNN) dan Logistic Regression mencatat akurasi yang lebih rendah, yaitu 82.52% dan 81.28%, menandakan bahwa model-model ini kurang optimal dalam menangkap pola churn pada dataset ini.

Adapun Decision Tree Classifier, meskipun mudah diinterpretasikan, hanya mencapai akurasi 78.20%, yang menunjukkan keterbatasan model ini dalam memodelkan kompleksitas data tanpa dukungan teknik ensemble.

## Hubungan dengan Business Understanding

1. Apakah sudah menjawab setiap problem statement?
    - Ya. Problem statement dalam laporan ini adalah:
    “Berdasarkan data historis pelanggan, dapatkah kita membangun sebuah model machine learning yang mampu memprediksi apakah seorang pelanggan akan keluar (churn) atau tidak?”
    - Model yang dikembangkan—khususnya Gradient Boosting Classifier dan Random Forest Classifier—berhasil memprediksi churn dengan akurasi tinggi (masing-masing sekitar 86.84% dan 86.64%).
    - Dengan demikian, model berhasil menjawab problem statement secara langsung, karena mampu melakukan klasifikasi churn dengan performa yang solid dan relevan untuk aplikasi nyata.

2. Apakah berhasil mencapai setiap goals yang diharapkan?
    - Ya. Terdapat tiga goals utama yang diuraikan, dan semuanya telah tercapai:
        - Identifikasi Faktor Risiko:
            - Model berhasil mengungkap fitur-fitur penting seperti CreditScore, Age, Balance, dan IsActiveMember sebagai prediktor utama.
            - Ini berarti tujuan untuk mengidentifikasi variabel-variabel yang berpengaruh sudah tercapai.
        - Pembangunan Model Prediktif:
            - Beberapa model klasifikasi dibangun dan dievaluasi.
            - Model terbaik memiliki performa yang tinggi dalam hal akurasi, precision, recall, dan F1-score.
        - Dukungan Keputusan Bisnis:
            - Output model dapat digunakan oleh tim bisnis untuk menargetkan pelanggan berisiko churn.
            - Potensi implementasi dalam sistem pendukung keputusan dijelaskan secara eksplisit.
    - Dengan demikian, semua goals yang didefinisikan telah terpenuhi dan dihubungkan kembali ke konteks bisnis.

3. Apakah setiap solution statement yang kamu rencanakan berdampak? Jelaskan!
    - Ya, solusinya memberikan dampak nyata.
    - Solution statement menyebutkan bahwa proyek akan melibatkan:
        * Pembersihan dan preprocessing data.
        * Pembangunan pipeline machine learning.
        * Evaluasi komprehensif menggunakan berbagai metrik.
        * Integrasi model ke sistem keputusan bisnis.
    - Dampak nyata dari solusi tersebut:
        - Pipeline data end-to-end telah dibangun, dari preprocessing hingga evaluasi.
        - Berbagai model telah dicoba, dan performa masing-masing dievaluasi secara objektif.
        - Model terbaik dapat diintegrasikan ke dalam sistem bisnis untuk membantu strategi retensi pelanggan.
        - Aspek bisnis dijawab langsung lewat analisis hasil: bank dapat menggunakan model ini untuk mengurangi churn, meningkatkan loyalitas pelanggan, dan mengoptimalkan alokasi sumber daya.



## Kesimpulan

Evaluasi terhadap beberapa model klasifikasi menunjukkan bahwa:

- Model Terbaik: Gradient Boosting Classifier, dengan akurasi tertinggi sebesar 86.84%, serta metrik evaluasi lainnya yang seimbang.
- Model Alternatif yang Kompetitif: Random Forest Classifier menunjukkan performa yang hampir sebanding, dan dapat dijadikan cadangan dalam sistem ensemble atau benchmarking.
- Model Lain: AdaBoost, KNN, Decision Tree, dan Logistic Regression memiliki performa yang layak, namun tidak melampaui dua model teratas.

Secara keseluruhan, proyek ini berhasil membangun sistem prediktif churn yang andal dan aplikatif untuk keperluan bisnis. Hasil ini dapat dimanfaatkan oleh Bank XYZ untuk meningkatkan strategi retensi pelanggan berbasis data dan mendukung pengambilan keputusan secara lebih tepat sasaran.


----
