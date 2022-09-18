# Laporan Proyek Machine Learning - Ades Tikaningsih
 

### Domain Proyek : Implementasi Hyperparameter Tuning Algoritma K-Nearest Neighborh untuk Prediksi Tanaman 

**Latar Belakang**

Pertanian merupakan salah satu sektor industri yang menjadi bagian pekerjaan juga pemenuhan kebutuhan masyarakat seperti kebutuhan makanan pokok. Salah satu hal penting dalam bidang pertanian adalah memilih jenis tanaman dengan mempertimbangkan beberapa faktor  seperti sifat tanah, kondisi iklim dan jenis pupuk yang digunakan, sering kali kita menghadapi perubahan iklim yang tidak terduga seperti curah hujan non-musim, gelombang panas atau fluktuasi tingkat kelembaban. Sebagai petani harus bisa mengelola dan menentukan skala yang ideal untuk pertumbuhan dan berkembangnya tanaman sehingga kerugian dapat diminimalisir. Oleh karena itu, perlu upaya untuk membantu petani dalam mengidentifikasi karakteristik tanaman sesuai dengan kondisi tanah, suhu, curah hujan dan pupuk NPK yaitu dengan memanfaatkan perkembangan teknologi khususnya bidang pertanian dibuatlah inovasi dari teknologi machine learning yaitu memprediksi kriteria tanaman terbaik agar dapat dapat membantu petani agar dapat menanam tanaman sesuai dengan karakteristik dan kondisi lingkungan. 
## Business Understanding

Petani ialah orang yang pekerjaannya bercocok tanam. Banyak keahlian yang dituntut untuk dimiliki oleh seorang petani, salah satunya mengidentifikasi pengelolaan tanah, kondisi cuaca, suhu dan pupuk. Upaya itu dilakukan agar petani mengetahui tanaman yang cocok sesuai karakteristiknya. Contoh sebagian besar tanaman padi membutuhkan air dalam jumlah besar, tetapi ada beberapa tanaman yang membutuhkan sedikit air. Pemberian air yang terlalu sedikit akan menyebabkan daun terkulai, dan terlalu banyak air akan menyebabkan akar membusuk. Oleh karena itu, di perlukan suatu metode untuk memprediksi tanaman terbaik dengan mempertimbangkan beberapa faktor.

**Problem Statement**

Berdasarkan kondisi yang telah diuraikan sebelumnya, proyek ini akan mengembangkan sebuah sistem prediksi rekomendasi tanaman terbaik berdasarkan kondisi curah hujan, pH tanah, kelembaban, suhu, dan pupuk NPK (Nitrogen, Fosfor,Kalium) dimana masing-masing elemen memiliki peran penting dalam pertumbuhan tanaman seperti :
* N (nitrogen) berfungsi pertumbuhan daun pada tanaman
*	P (fosfor) untuk pertumbuhan akar dan perkembangan bunga dan buah
*	K (Kalium) berfungsi untuk keseluruhan tanaman agar berfungsi dengan benar
*	Suhu atau temperatur mempunyai pengaruh terhadap metabolisme, fotosintesis, dan transpirasi tumbuhan. 
*	Kelembaban berfungsi pada beberapa tanaman agar tubuhnya tidak cepat kering karena penguapan
*	Ph tanah berfungsi dalam jumlah tertentu untuk tumbuh, berkembang, dan bertahan terhadap penyakit
*	Curah hujan dapat membantu tumbuhan dalam bertumbuh dengan bantuan air.

Dengan menggunakan teknologi machine learning algoritma K-Nearest Neighborh diharapkan dapat menjawab permasalahan berikut :
- Bagaimana menentukkan tanaman yang cocok sesuai dengan kondisi curah hujan, pH tanah, suhu, kelembabn dan pupuk NPK ?
- Bagaimana menentukkan nilai 'k' terbaik pada algoritma K-Nearest Neighbor.

**Goal**

Untuk  menjawab problem statement tersebut, akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:
- Membuat model machine learning yang dapat memprediksi tanaman yang cocok berdasarkan parameter yang di tetapkan.
- Mengetahui nilai 'k' terbaik menggunakan Hyperparameter Tuning Algoritma KNN.

**Solution Statement**

Untuk mencapai goals tersebut, proyek ini menggunakan salah satu algoritma Machine Learning yaitu K-Nearest Neighborh. Konsep dasar dari KNN adalah melihat berapa banyak ‘tetangga’ yang paling dekat dengan data baru yang hendak diprediksi datanya. K pada nama KNN disini menunjukan ‘tetangga’ maksimal untuk menentukan dikategori mana data baru ini akan terprediksi untuk memperoleh hasil akurasi yang maksimal, kita tentukan K sementara terlebih dahulu. Karena setelah ini akan dilakukan proses optimalisasi model dengan memilih parameter terbaik (hyperparameter tuning) sehingga K ini bersifat sementara. Setelah itu akan dibuat perbandingan berdasarkan nilai akurasi yang dihasilkan sebelum dan sesudah menggunakan hyperparameter tuning. Hasil akurasi  terbaiklah yang digunakan untuk memprediksi.

## Data Understanding

Data yang akan digunakan pada proyek kali ini adalah Crop Recommendation Dataset India yang diunduh dari  [kaggle](https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset) Jumlah sample data 2200 records dan 8 kolom. Berikut ini karakteristik dataset : 
- N - rasio kandungan Nitrogen dalam tanah
- P - rasio kandungan Fosfor dalam tanah
- K - rasio kandungan Kalium dalam tanah
- suhu - suhu dalam derajat Celcius
- kelembaban - kelembaban relatif dalam %
- ph - nilai ph tanah
- curah hujan - curah hujan dalam mm

Untuk memahami data akan dilakukan proses berikut ini :

**1. Data Loading**

Supaya isi dataset lebih mudah dipahami, kita perlu melakukan proses loading data terlebih dahulu dengan import library pandas untuk dapat membaca file datanya. 

**2. Exploratory Data Analysis (EDA)**

###### Informasi Dataset
mengecek informasi pada dataset dengan fungsi info() berikut. 

![gambar](https://github.com/adstika20/submission_predictive_analytics/blob/main/data.info().png)

Berdasarkan informasi diatas dataset memiliki beberapa kriteria antara lain :
*   4 Kolom dengan tipe float64 yaitu temperature, humidity, ph, rainfall
*   3 Kolom dengan tipe int64 yaitu N, P, K
*   1 Kolom dengan tipe object yaitu label

###### Cek Missing Value
Jika data terdiri dari ratusan bahkan ribuan baris tentu akan susah dalam menemukan nilai field yang kosong. Oleh karena itu, Pandas memungkinkan kita dapat menemukan missing value secara cepat dengan fungsi isna() dan sum().

![missing value](https://github.com/adstika20/submission_predictive_analytics/blob/main/missing%20value.png)

Pada dataset crop recommendation tidak ada missing value, sehingga bisa dilanjutkan untuk proses berikutnya.
###### Menangani Outlier
Beberapa pengamatan dalam satu set data kadang berada di luar lingkungan pengamatan lainnya. Pengamatan seperti itu disebut outlier. Dampak jika tidak menanggulangi nilai outliers akan menyebabkan ketidakakuratan model machine learning yang dikerjakan. Berikut visualisasi beberapa data outlier 

![](https://github.com/adstika20/submission_predictive_analytics/blob/main/Outlier.png)

Untuk mengatasi outlier menggunakan metode IQR dengan mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier. Hal pertama yang perlu Anda lakukan adalah membuat batas bawah dan batas atas. Untuk membuat batas bawah, kurangi Q1 dengan 1,5 * IQR. Kemudian, untuk membuat batas atas, tambahkan 1.5 * IQR dengan Q3. Setelah menghilangkan outliers cek kembali data. Data telah bersih dan memiliki 1768 sampel.
###### Bivariate Analysis
Bivariate Analysis dilakukan untuk menemukan hubungan antara setiap variabel dalam kumpulan data dan variabel target yang diinginkan (atau) menggunakan 2 variabel dan menemukan hubungan di antara keduanya.

![](https://github.com/adstika20/submission_predictive_analytics/blob/main/bivariate%20analysis.png) 
![](https://github.com/adstika20/submission_predictive_analytics/blob/main/bivariate1.png)
![](https://github.com/adstika20/submission_predictive_analytics/blob/main/bivariate2.png)

Figure diatas dapat memberi tahu kita variabel kategori tersebut sangat sebagian besar seimbang, sehingga diagram korelasi di atas dapat disimpulkan sebagai berikut :
*   Kapas,pisang,semangka,kopi,  membutuhkan sebagian besar Nitrogen
*   Pisang, membutuhkan sebagian besar Fosfor
*   Buncis membutuhkan sebagian besar Kalium
*   Banana, membutuhkan sebagian besar Fosfor
*   Sebagian besar tanaman membutuhkan iklim yang panas
*   Sebagian besar tanaman membutuhkan iklim yang lembab kecuali buncis
*   Sebagian besar tanaman  membutuhkan pH tinggi di tanah.
*   Padi membutuhkan sebagian besar curah hujan yang sangat besar

## Data Preparation
Pada bagian ini akan melakukan beberapa tahap persiapan data, yaitu:
###### 1. Membagi Feature dan Target
Bagi fitur pada dataset menjadi dua bagian, pertama numerical features terdiri dari 'N','P','K','temperature','humidity','ph','rainfall'. Kedua target terdiri dari semua label dataset.
###### 2. Train test split
Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. proporsi pembagian data latih dan uji adalah 80:20. Proporsi tersebut cukup ideal untuk model dengan jumlah data 1768. Namun,  jika memiliki dataset berukuran besar, kita perlu memikirkan strategi pembagian dataset lain agar proporsi data uji tidak terlalu banyak. Pembagian ini menggunakan fungsi train_test_split dari sklearn hasil yang diperoleh berikut :
```sh
The Shape of x train: (1414, 7)
The Shape of x test: (354, 7)
The Shape of y train: (1414,)
The Shape of y test: (354,)
```
###### 3. Standarisasi
Proses standarisasi menggunakan fungsi StandardScaler. Fungsi StandardScaler() akan menormalkan fitur-fitur (setiap kolom X) sehingga setiap kolom/variabel akan memiliki mean = 0 dan standard deviation = 1.

![](https://github.com/adstika20/submission_predictive_analytics/blob/main/standarscaler.png)

Sampai di tahap ini, data kita telah siap untuk dilatih menggunakan model machine learning
## Modeling
Untuk menyelesaikan proyek ini menggunakan Algoritma K-nearest Neighbor dengan menentukan nilai 'k' terbaik menggunakan hyperparameter tuning.Pengukuran performa model didasarkan pada akurasinya. Berikut tahapan membuat model dengan KNN :
**1. Konsep Dasar K Nearest Neighbor**
Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). 
**2. Menentukkan k terbaik (sementara)**
Untuk memperoleh hasil akurasi yang maksimal, tentukan K sementara terlebih dahulu. K inisialisasi ini akan menjadi K pada model awal sebelum optimalisasi (tuning) yang nantinya akan dibandingkan nilai performanya dengan model setelah tuning.Pada proyek ini menggunakan nilai k dengan range 1 sampai 50 interval 2 
```sh
1,3,5,7....49
```
Jika melihat output diatas, dapat disimpulkan bahwa K (sementara) yang terbaik adalah K = 49. Nilai akurasinya mencapai 91.24%.  Akurasi tersebut cukup bagus tetapi mari mengecek matrik selain nilai akurasi, seperti precision, recall, dan f1-score, dengan menerapkan fungsi classification_report dari library sklearn. Presisi idealnya harus bernilai 1 (tinggi) untuk pengklasifikasi yang baik, begitu juga dengan recall. Kemdudian Skor F1 menjadi 1 hanya jika presisi dan daya ingat keduanya 1. Skor F1 menjadi tinggi hanya jika presisi dan daya ingat tinggi. Skor F1, presisi dan recall dan merupakan ukuran yang lebih baik daripada melihat nilai akurasi dalam menentukkan performa model.

![](https://github.com/adstika20/submission_predictive_analytics/blob/main/matriks_knn_before.png)

Berdasarkan gambar hampir semua kelas, akurasi precision, recall, dan f1-score memiliki angka yang tinggi, rata-rata di atas 90-an. Namun, pada beberapa kelas seperti kelas dengan indeks 11 dan 18, akurasi rata-ratanya berada dibawah 50%. Hal ini bisa disebabkan oleh representasi fitur yang cukup kompleks dari kedua kelas ini. Oleh karena itu, mari kita menerapkan hyperparameter tuning untuk memperoleh hasil yang lebih baik.
**3. Hyperparameter Tuning**
Hyperparameter tuning adalah sebuah proses untuk melakukan optimalisasi parameter pada sebuah model. Dalam KNN, terdapat beberapa parameter yang menjadi pembangun model. K atau dikenal dengan ‘n_neighbors’ merupakan salah satu parameter yang sudah dikenalkan diawal. Untuk proyek ini Parameter yang di gunakan ada 3 yaitu :
*   n_neighbors : menentukan nilai k terbaik berdasarkan nilai yang telah dihitung
*   Weight (bobot) : periksa penambahan bobot seragam atau jarak
*   Metrics : Jarak yang digunakan untuk menghitung kesamaan

Mula-mula tentukan terlebih dahulu rata-rata akurasi dengan fungsi np.zero() akurasi rata-rata dari 20 eksperimen pelatihan sebagai berikut.
```sh
array([0.98305085, 0.97175141, 0.97175141, 0.97740113, 0.97457627,
       0.97457627, 0.97175141, 0.96892655, 0.96892655, 0.96892655,
       0.96892655, 0.96892655, 0.97175141, 0.97175141, 0.97175141,
       0.97175141, 0.97175141, 0.95762712, 0.95762712, 0.95762712])
```
Ini berarti bahwa ketika sampai pada kumpulan sampel independen lainnya (set pengujian), diharapkan dapat mencapai kisaran akurasi tersebut. 

![](https://github.com/adstika20/submission_predictive_analytics/blob/main/matriks_knn.png)

Selanjutnya menentukkan kandidat untuk memilih parameter terbaik 
```sh
{ 'n_neighbors' : [12,13,14,15,16,17,18],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
```
Kemudian dengan menggunakan GridSearchCV Scikit-Learn untuk mencari parameter terbaik yang dilakukan secara brute force dan melaporkan mana parameter yang memiliki akurasi paling baik.
Output dibawah ini menjelaskan bahwa parameter terbaik untuk nilai 'k' adalah 13 , metric : manhattan dan weights : distance.
Kemudian melakukan fit model dengan menerapkan nilai k terbaik sebelumnya yang diperoleh hasil berikut :


## Evaluation
Bandingkan dengan model setelah dilakukan optimalisasi dengan tuning.Selain akurasi yang didapatkan juga meningkat menjadi 98%. Selain itu, berdasarkan klasifikasi report rata-rata Nilai terbaik F1-Score adalah 1.0 dan nilai terburuknya adalah 0.67. Secara representasi, setelah menggunakan hyperparameter metrik precision, recall, dan f1-score dominan menghasilkan akurasi terbaik 1.0. Sehingga dapat dikatakan bahwa dengan menggunakan hyperparameter tuning ini dapat meningkatkan nilai performa model yang digunakan. 
 gambar.....
Dengan kondisi N = 83, P = 45, K = 60, temperature = 28 C, humidity = 70.3 %, ph = 7.0, dan rainfall = 150 mm tanaman yang cocok di adalah jute atau daun rami.


## References

