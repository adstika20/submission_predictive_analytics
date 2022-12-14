# Laporan Proyek Machine Learning - Ades Tikaningsih
 

## Domain Proyek

**Latar Belakang**

Pertanian merupakan salah satu sektor industri yang menjadi bagian pekerjaan juga pemenuhan kebutuhan  kebutuhan makanan pokok bagi masyarakat. Salah satu hal penting dalam bidang pertanian adalah memilih jenis tanaman dengan mempertimbangkan faktor cuaca, suhu, tanah dan jenis pupuk yang digunakan. Sering kali kita menghadapi perubahan iklim yang tidak terduga seperti curah hujan non-musim, gelombang panas atau fluktuasi tingkat kelembaban. Sebagai petani harus bisa mengelola dan menentukan skala yang ideal untuk pertumbuhan dan berkembang tanaman sehingga kerugian dapat diminimalisir. Petani dapat menghasilkan tanaman sepanjang tahun dan kesuburan tanah tetap terjaga dengan memperhatikan kualitas tanah yang ditentukan oleh nilai NPK dari tanah. 'N' adalah kandungan nitrogen tanah, 'P' adalah kandungan fosfor dan 'K' adalah kandungan kalium dari tanah.[[1](https://www.researchgate.net/profile/Hem-Joshi-3/post/what_are_the_different_techniques_used_in_fertilizers_crop_recommendation_system/attachment/5f08a4a43f16f90001231756/AS%3A911802045038593%401594401956134/download/ijcsit2016070247.pdf)]. Oleh karena itu, perlu upaya untuk membantu petani dalam mengidentifikasi karakteristik tanaman sesuai dengan kondisi tanah, suhu, curah hujan dan pupuk NPK yaitu dengan memanfaatkan perkembangan teknologi khususnya bidang pertanian dibuatlah inovasi dari teknologi machine learning yaitu memprediksi kriteria tanaman terbaik agar dapat dapat membantu petani dalam menanam tanaman sesuai dengan karakteristik dan kondisi lingkungan.

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

Dengan menggunakan teknologi machine learning algoritma K-Nearest Neighbor dan hyperparameter tuning diharapkan dapat menjawab permasalahan berikut :
- Bagaimana menentukkan tanaman yang cocok sesuai dengan kondisi curah hujan, pH tanah, suhu, kelembabn dan pupuk NPK ?
- Bagaimana menentukkan nilai 'k' terbaik pada algoritma K-Nearest Neighbor menggunakan teknik hyperparameter tuning.

**Goal**

Untuk  menjawab problem statement tersebut, akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:
- Membuat model machine learning yang dapat memprediksi tanaman yang cocok berdasarkan parameter yang di tetapkan.
- Mengetahui nilai 'k' terbaik menggunakan Hyperparameter Tuning Algoritma KNN.

**Solution Statement**

Untuk mencapai goals tersebut, proyek ini menggunakan salah satu algoritma Machine Learning yaitu K-Nearest Neighborh. Konsep dasar dari KNN adalah melihat berapa banyak ???tetangga??? yang paling dekat dengan data baru yang hendak diprediksi datanya. K pada nama KNN disini menunjukan ???tetangga??? maksimal untuk menentukan dikategori mana data baru ini akan terprediksi untuk memperoleh hasil akurasi yang maksimal, kita tentukan K sementara terlebih dahulu. Karena setelah ini akan dilakukan proses optimalisasi model dengan memilih parameter terbaik (hyperparameter tuning) sehingga K ini bersifat sementara. Setelah itu akan dibuat perbandingan berdasarkan nilai akurasi yang dihasilkan sebelum dan sesudah menggunakan hyperparameter tuning. Hasil akurasi terbaiklah yang digunakan untuk memprediksi.

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

![image](https://user-images.githubusercontent.com/110407053/191157940-16f0c906-8600-470f-b877-6befa0769503.png)

Berdasarkan informasi diatas dataset memiliki beberapa kriteria antara lain :
*   4 Kolom dengan tipe float64 yaitu temperature, humidity, ph, rainfall
*   3 Kolom dengan tipe int64 yaitu N, P, K
*   1 Kolom dengan tipe object yaitu label

###### Cek Missing Value
Jika data terdiri dari ratusan bahkan ribuan baris tentu akan susah dalam menemukan nilai field yang kosong. Oleh karena itu, Pandas memungkinkan kita dapat menemukan missing value secara cepat dengan fungsi isna() dan sum().

![image](https://user-images.githubusercontent.com/110407053/191158093-4d3f3239-79e4-448e-8d8f-6156a2dce413.png)

Pada dataset crop recommendation tidak ada missing value, sehingga bisa dilanjutkan untuk proses berikutnya.
###### Menangani Outlier
Beberapa pengamatan dalam satu set data kadang berada di luar lingkungan pengamatan lainnya. Pengamatan seperti itu disebut outlier. Dampak jika tidak menanggulangi nilai outliers akan menyebabkan ketidakakuratan model machine learning yang dikerjakan. Berikut visualisasi beberapa data outlier 

![image](https://user-images.githubusercontent.com/110407053/191158147-78fa4ece-1b93-4c99-aabf-b6b3e1f82162.png)

Untuk mengatasi outlier menggunakan metode IQR dengan mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier. Hal pertama yang perlu Anda lakukan adalah membuat batas bawah dan batas atas. Untuk membuat batas bawah, kurangi Q1 dengan 1,5 * IQR. Kemudian, untuk membuat batas atas, tambahkan 1.5 * IQR dengan Q3. Setelah menghilangkan outliers cek kembali data. Data telah bersih dan memiliki 1768 sampel.
###### Bivariate Analysis
Bivariate Analysis dilakukan untuk menemukan hubungan antara setiap variabel dalam kumpulan data dan variabel target yang diinginkan (atau) menggunakan 2 variabel dan menemukan hubungan di antara keduanya.

![image](https://user-images.githubusercontent.com/110407053/191158201-54920db9-9e53-4e14-abe0-6bbbc834a20f.png)
![image](https://user-images.githubusercontent.com/110407053/191158230-6aaeffbe-c2b3-4772-8e1e-79062d4d2690.png)
![image](https://user-images.githubusercontent.com/110407053/191158254-c8215e05-2369-4da0-93ba-3e3c3fc024df.png)


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
## Modeling
Untuk menyelesaikan proyek ini menggunakan Algoritma K-nearest Neighbor dengan menentukan nilai 'k' terbaik menggunakan hyperparameter tuning.Pengukuran performa model didasarkan pada akurasinya. Berikut tahapan membuat model dengan KNN :

**1. Konsep Dasar K Nearest Neighbor**

Algoritma KNN menggunakan ???kesamaan fitur??? untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). 

**2. Menentukkan k terbaik (sementara)**

Untuk memperoleh hasil akurasi yang maksimal, tentukan K sementara terlebih dahulu. K inisialisasi ini akan menjadi K pada model awal sebelum optimalisasi (tuning) yang nantinya akan dibandingkan nilai performanya dengan model setelah tuning berdasarkan akurasi yang diperoleh. Pada proyek ini menggunakan nilai k dengan range 1 sampai 50 interval 2 

![image](https://user-images.githubusercontent.com/110407053/191158303-37997fec-0753-4f25-bbf7-ad0e26c3fc25.png)


Jika melihat output diatas, dapat disimpulkan bahwa K (sementara) yang terbaik adalah K = 49. Nilai akurasinya mencapai 91.24%.  Akurasi tersebut cukup bagus tetapi mari mengecek matrik selain nilai akurasi, seperti precision, recall, dan f1-score, dengan menerapkan fungsi classification_report dari library sklearn. Presisi idealnya harus bernilai 1 (tinggi) untuk pengklasifikasi yang baik, begitu juga dengan recall. Kemudian skor F1 menjadi 1 hanya jika presisi dan recall keduanya 1. Skor F1, presisi dan recall dan merupakan ukuran yang lebih baik daripada melihat nilai akurasi dalam menentukkan performa model.

![image](https://user-images.githubusercontent.com/110407053/191158354-583aa8c8-0092-4fab-9a44-802c23b7dae9.png)

Berdasarkan gambar diatas hampir semua kelas, akurasi precision, recall, dan f1-score memiliki angka yang tinggi, rata-rata di atas 90-an. Namun, pada beberapa kelas seperti kelas dengan indeks 11 dan 18, akurasi rata-ratanya berada dibawah 50%. Hal ini bisa disebabkan oleh representasi fitur yang cukup kompleks dari kedua kelas ini. Oleh karena itu, mari kita menerapkan hyperparameter tuning untuk memperoleh hasil yang lebih baik.

**3. Hyperparameter Tuning**

Hyperparameter tuning adalah sebuah proses untuk melakukan optimalisasi parameter pada sebuah model. Dalam KNN, terdapat beberapa parameter yang menjadi pembangun model. K atau dikenal dengan ???n_neighbors??? merupakan salah satu parameter yang sudah dikenalkan diawal. Untuk proyek ini Parameter yang di gunakan ada 3 yaitu :
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
Selanjutnya menentukkan kandidat untuk memilih parameter terbaik dengan ketentuan berikut :
* n_neighbors : 12,13,14,15,16,17,18
* weights : 'uniform','distance'
* metric : 'minkowski','euclidean','manhattan'

Kemudian dengan menggunakan GridSearchCV Scikit-Learn untuk mencari parameter yang dilakukan secara brute force dan melaporkan mana parameter yang memiliki akurasi paling baik. Setelah dilakukan proses pencarian parameter yang optimal menggunakan GridSearch diperoleh parameter nilai 'k' adalah 13 , metric : manhattan dan weights : distance yang akan digunakan untuk melakukan fit model yang diperoleh hasil berikut :

![image](https://user-images.githubusercontent.com/110407053/191158404-cab3fa91-58a6-4123-ac41-7b50011e83c9.png)

 Nilai akurasi model meningkat setelah diterapkan hyperparameter tuning dengan perolehan nilai train 100 % dan test 97 %. Tentunya performa model lebih baik jika dibandingkan dengan akurasi sebelum dilakukan tuning.

## Evaluation
Seperti yang telah dijelaskan sebelumnya matrik evaluasi yang digunakan adalah akurasi, precision,recall dan F1 score. Dimana precision idealnya harus bernilai 1 (tinggi) untuk pengklasifikasi yang baik, begitu juga dengan recall. Kemudian skor F1 menjadi 1 hanya jika presisi dan recall keduanya 1. Skor F1, presisi dan recall dan merupakan ukuran yang lebih baik daripada melihat nilai akurasi dalam menentukkan performa model.

Jika melihat report dari model sebelum proses tuning, nilai akurasinya mencapai 91%. Meski akurasi yang dihasilkan adalah 91%, namun berdasarkan fungsi classification report() nilai precission, recall, dan f1-score recall pada beberapa kelas hanya sebesar 30% . Artinya prediksi pada kelas masih banyak kesalahan. Bandingkan dengan model setelah dilakukan optimalisasi dengan tuning. Selain akurasi yang didapatkan juga meningkat menjadi train 100 % dan test 97 %. Rata-rata Nilai terbaik F1-Score adalah 1.0 dan nilai terburuknya adalah 0.67. 

![image](https://user-images.githubusercontent.com/110407053/191158448-ad5910d7-ec1d-4d57-88ff-c104f94af935.png)

Sehingga dapat disimpulkan dengan menggunakan hyperparameter tuning ini dapat meningkatkan nilai performa model yang digunakan. Meski begitu, kekurangan dari tuning ini adalah semakin banyak data dan semakin banyak parameter yang hendak diuji, maka semakin lama waktu proses yang dibutuhkan. 

![image](https://user-images.githubusercontent.com/110407053/191158500-205e118f-4e69-41e3-a1ba-1288116dbd96.png)

Dengan kondisi N = 83, P = 45, K = 60, temperature = 28 C, humidity = 70.3 %, ph = 7.0, dan rainfall = 150 mm tanaman yang cocok di adalah jute atau daun rami. Dengan menggunakan Hyperparameter Tuning untuk mencari nilai k pada algoritma k-nearest neighbor dan beberapa metrik lain dapat meningkatkan performa model dan dapat memprediksi tanaman sesuai dengan karakteristiknya.

## References

[[1](https://www.researchgate.net/profile/Hem-Joshi-3/post/what_are_the_different_techniques_used_in_fertilizers_crop_recommendation_system/attachment/5f08a4a43f16f90001231756/AS%3A911802045038593%401594401956134/download/ijcsit2016070247.pdf)] S. Mansi , E. Kimaya , G. Sonali , P. Sanket and M. Shubhada , "Crop Recommendation and Fertilizer Purchase System," (IJCSIT) International Journal of Computer Science and Information Technologies, vol. 7, no. 2, 2016. 
