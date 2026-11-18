# Pengenalan Pola dan Machine Learning: Pendekatan Probabilistik dengan Python

Kode dan notebook pendamping buku *Pengenalan Pola dan Machine Learning:
Pendekatan Probabilistik dengan Python* karya Lestari Handayani dan Fitri Insani
(Nalar Press Indonesia, 2027).

Repositori ini berisi semua listing di buku: 75 listing Python di 18 bab, satu
notebook Jupyter per bab yang dapat langsung dijalankan di Google Colab atau
Kaggle, dan keluaran yang dicetak di buku untuk pembanding.

## Isi repositori

```
kode/babNN/            skrip setiap listing, ditambah modul data_*.py
kode/babNN/keluaran/   keluaran setiap skrip seperti tercetak di buku
notebook/babNN.ipynb   semua listing satu bab dalam satu notebook
cek_keluaran.py        menjalankan semua skrip dan membandingkan keluarannya
requirements.txt       pustaka yang diperlukan
```

Semua data di buku ini **sintetis**: dibangkitkan oleh modul `data_*.py` di
folder bab masing-masing dengan benih acak tetap, sehingga tidak perlu
mengunduh apa pun. Konteksnya (curah hujan, harga cabai, stunting, IPM
kabupaten, bantuan sosial, dan sebagainya) dipilih agar dekat dengan
Indonesia, tetapi angkanya ilustrasi, bukan data resmi.

## Cara menjalankan

### Di Google Colab atau Kaggle (tanpa memasang apa pun)

Klik tombol **Open in Colab** pada bab yang diinginkan di daftar di bawah, lalu
pilih *Runtime → Run all*. Sel pertama setiap notebook menulis skrip bab itu
ke folder kerja, jadi notebook dapat dijalankan sendiri tanpa menyalin
repositori. Bab 15 memasang PyMC di sel kedua (sekitar satu menit). Untuk
Kaggle, unggah berkas `.ipynb` lewat *File → Import Notebook*.

### Di komputer sendiri

```bash
git clone https://github.com/bsnegara/pola-ml-python.git
cd pola-ml-python
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd kode/bab01
python polinomial_hujan.py
```

Jalankan setiap skrip **dari foldernya sendiri**, karena skrip mengimpor modul
data di folder yang sama. Semua kode berjalan tanpa GPU. Kebanyakan selesai
dalam hitungan detik; beberapa skrip di Bab 15–18 memerlukan beberapa menit.
`kode/bab05/laplace_logistik.py` memakai hasil `logistik_irls.py`, jadi
jalankan yang kedua lebih dulu.

### Memeriksa keluaran

```bash
python cek_keluaran.py          # semua bab (sekitar 20–30 menit)
python cek_keluaran.py 4 16     # hanya Bab 4 dan 16
```

Angka di buku dihasilkan dengan Python 3.13, NumPy 2.5, scikit-learn 1.9,
PyMC 6 dan PyTorch 2.15 di CPU. Dengan versi lain, angka di digit terakhir bisa
sedikit berbeda, terutama pada skrip yang memakai optimisasi atau PyTorch.
Baris yang mencatat waktu eksekusi selalu berbeda. Kesimpulan yang dibahas di
buku tidak bergantung pada selisih seperti itu.

## Daftar listing per bab

### Bab 1. Pengenalan Pola sebagai Inferensi

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab01.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 1.1 | Mencocokkan polinom pada data curah hujan sintetis | [`polinomial_hujan.py`](kode/bab01/polinomial_hujan.py) |
| 1.2 | Bias taksiran maximum likelihood untuk varians derau | [`mle_derau.py`](kode/bab01/mle_derau.py) |
| 1.3 | Estimasi MAP: prior Gaussian sebagai regularisasi | [`map_ridge.py`](kode/bab01/map_ridge.py) |
| 1.4 | Distribusi prediktif Bayesian untuk regresi polinom | [`prediksi_bayes.py`](kode/bab01/prediksi_bayes.py) |

### Bab 2. Probabilitas, Keputusan, dan Informasi

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab02.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 2.1 | Teorema Bayes untuk skrining DBD, diperiksa dengan simulasi | [`bayes_dbd.py`](kode/bab02/bayes_dbd.py) |
| 2.2 | Dua gejala kutukan dimensi | [`kutukan_dimensi.py`](kode/bab02/kutukan_dimensi.py) |
| 2.3 | Tiga aturan keputusan untuk rujukan DBD | [`keputusan_biaya.py`](kode/bab02/keputusan_biaya.py) |
| 2.4 | Entropi distribusi huruf dan divergensi KL | [`entropi_teks.py`](kode/bab02/entropi_teks.py) |

### Bab 3. Distribusi Probabilitas

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab03.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 3.1 | Prevalensi stunting dua desa: maximum likelihood dan posterior beta | [`beta_stunting.py`](kode/bab03/beta_stunting.py) |
| 3.2 | Berat badan bersyarat tinggi badan pada Gaussian dua dimensi | [`gaussian_bersyarat.py`](kode/bab03/gaussian_bersyarat.py) |
| 3.3 | Memeriksa bias varians maximum likelihood dengan simulasi | [`bias_varians.py`](kode/bab03/bias_varians.py) |
| 3.4 | Pembaruan Bayesian sekuensial untuk rerata tinggi badan | [`gauss_sekuensial.py`](kode/bab03/gauss_sekuensial.py) |
| 3.5 | Estimasi kerapatan kernel untuk tinggi badan, dengan $h$ dipilih dengan data validasi | [`kde_tinggi.py`](kode/bab03/kde_tinggi.py) |

### Bab 4. Regresi Linear dan Bayesian

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab04.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 4.1 | Membandingkan fungsi basis untuk harga cabai mingguan | [`basis_harga.py`](kode/bab04/basis_harga.py) |
| 4.2 | Lasso sebagai pemilih fitur, diuji dengan fitur pengecoh | [`lasso_sparsitas.py`](kode/bab04/lasso_sparsitas.py) |
| 4.3 | Mengukur bias dan variansi dengan 100 set data | [`bias_variansi.py`](kode/bab04/bias_variansi.py) |
| 4.4 | Regresi linear Bayesian untuk harga cabai, dengan $$ dan $$ dari data | [`bayes_harga.py`](kode/bab04/bayes_harga.py) |
| 4.5 | Evidence dan parameter efektif untuk orde polinom 0–9 | [`evidence_orde.py`](kode/bab04/evidence_orde.py) |

### Bab 5. Klasifikasi Linear

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab05.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 5.1 | Kuadrat terkecil dan diskriminan Fisher pada bag of words SMS | [`kuadrat_fisher.py`](kode/bab05/kuadrat_fisher.py) |
| 5.2 | Naive Bayes Bernoulli dari nol untuk SMS penipuan | [`naive_bayes_sms.py`](kode/bab05/naive_bayes_sms.py) |
| 5.3 | Regresi logistik dengan IRLS dan prior Gaussian, dari nol | [`logistik_irls.py`](kode/bab05/logistik_irls.py) |
| 5.4 | Regresi softmax untuk jenis penipuan | [`softmax_jenis.py`](kode/bab05/softmax_jenis.py) |
| 5.5 | Regresi logistik Bayesian dengan aproksimasi Laplace | [`laplace_logistik.py`](kode/bab05/laplace_logistik.py) |

### Bab 6. Evaluasi dan Pemilihan Model

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab06.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 6.1 | Akurasi, presisi, recall, dan F1 pada data tidak seimbang | [`akurasi_menipu.py`](kode/bab06/akurasi_menipu.py) |
| 6.2 | Dua bentuk kebocoran: seleksi fitur dan panah waktu | [`kebocoran.py`](kode/bab06/kebocoran.py) |
| 6.3 | ROC, presisi–recall, dan ambang berbasis biaya | [`roc_biaya.py`](kode/bab06/roc_biaya.py) |
| 6.4 | Undersampling dan koreksi prior | [`koreksi_prior.py`](kode/bab06/koreksi_prior.py) |
| 6.5 | Uji McNemar dan selang bootstrap untuk selisih kinerja | [`banding_model.py`](kode/bab06/banding_model.py) |

### Bab 7. Metode Kernel dan Proses Gaussian

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab07.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 7.1 | Ridge dalam bentuk primal dan dual memberi prediksi yang sama | [`dual_ridge.py`](kode/bab07/dual_ridge.py) |
| 7.2 | Regresi proses Gaussian dari nol untuk PM$_2,5$ per jam | [`gp_regresi.py`](kode/bab07/gp_regresi.py) |
| 7.3 | Marginal likelihood sebagai fungsi skala panjang | [`gp_hiperparameter.py`](kode/bab07/gp_hiperparameter.py) |
| 7.4 | Peta PM$_2,5$ dari 40 sensor dengan proses Gaussian | [`gp_spasial.py`](kode/bab07/gp_spasial.py) |

### Bab 8. Support Vector Machine

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab08.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 8.1 | Margin keras dan lunak untuk parang lawan kawung | [`svm_margin.py`](kode/bab08/svm_margin.py) |
| 8.2 | SVM RBF untuk empat motif, dengan pencarian kisi | [`svm_batik.py`](kode/bab08/svm_batik.py) |
| 8.3 | SVR dan relevance vector machine pada data sinc | [`svr_rvm.py`](kode/bab08/svr_rvm.py) |

### Bab 9. Pohon Keputusan dan Ensemble

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab09.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 9.1 | Pohon keputusan dengan berbagai kedalaman | [`pohon_kredit.py`](kode/bab09/pohon_kredit.py) |
| 9.2 | Satu pohon, bagging, dan random forest | [`ensemble_kredit.py`](kode/bab09/ensemble_kredit.py) |
| 9.3 | AdaBoost dengan decision stump, dari nol | [`adaboost_dari_nol.py`](kode/bab09/adaboost_dari_nol.py) |
| 9.4 | Gradient boosting dengan early stopping | [`boosting_kredit.py`](kode/bab09/boosting_kredit.py) |
| 9.5 | Kepentingan permutasi dan nilai Shapley eksak, dari nol | [`shap_dari_nol.py`](kode/bab09/shap_dari_nol.py) |

### Bab 10. Model Grafis

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab10.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 10.1 | Inferensi pada jaringan Bayes dengan enumerasi | [`jaringan_bayes.py`](kode/bab10/jaringan_bayes.py) |
| 10.2 | Membersihkan citra biner dengan MRF dan ICM | [`mrf_bersihkan.py`](kode/bab10/mrf_bersihkan.py) |
| 10.3 | Sum-product dan max-sum pada rantai 16 unit, dibanding enumerasi | [`sum_product_rantai.py`](kode/bab10/sum_product_rantai.py) |

### Bab 11. Klasterisasi, Model Campuran, dan EM

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab11.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 11.1 | K-means dari nol pada dua indikator IPM | [`kmeans_dari_nol.py`](kode/bab11/kmeans_dari_nol.py) |
| 11.2 | EM untuk campuran Gaussian, dari nol | [`em_gmm.py`](kode/bab11/em_gmm.py) |
| 11.3 | Memilih $K$ dan bentuk kovarians dengan BIC | [`pilih_k.py`](kode/bab11/pilih_k.py) |
| 11.4 | DBSCAN untuk titik panas sintetis | [`dbscan_titik_panas.py`](kode/bab11/dbscan_titik_panas.py) |

### Bab 12. Variabel Laten Kontinu

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab12.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 12.1 | PCA dari nol pada 38 provinsi sintetis | [`pca_provinsi.py`](kode/bab12/pca_provinsi.py) |
| 12.2 | PPCA bentuk tertutup dan analisis faktor | [`ppca_fa.py`](kode/bab12/ppca_fa.py) |
| 12.3 | Memisahkan dua sumber getaran dengan FastICA | [`ica_getaran.py`](kode/bab12/ica_getaran.py) |
| 12.4 | Kernel PCA pada dua cincin, dan t-SNE pada gulungan berdimensi 20 | [`nonlinear.py`](kode/bab12/nonlinear.py) |

### Bab 13. Data Sekuensial

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab13.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 13.1 | Rantai Markov hujan harian | [`rantai_markov.py`](kode/bab13/rantai_markov.py) |
| 13.2 | HMM untuk penandaan kelas kata: pelatihan, Viterbi, dan maju–mundur | [`hmm_pos.py`](kode/bab13/hmm_pos.py) |
| 13.3 | Filter Kalman dan penghalus RTS untuk pelacakan kapal | [`kalman_kapal.py`](kode/bab13/kalman_kapal.py) |

### Bab 14. Inferensi Variasional

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab14.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 14.1 | Mean-field untuk rerata dan presisi Gaussian | [`vb_gauss.py`](kode/bab14/vb_gauss.py) |
| 14.2 | EM dan inferensi variasional dengan sepuluh komponen untuk data empat kelompok | [`vb_campuran.py`](kode/bab14/vb_campuran.py) |
| 14.3 | Model topik LDA untuk berita daerah sintetis | [`lda_berita.py`](kode/bab14/lda_berita.py) |

### Bab 15. Metode Sampling dan Pemrograman Probabilistik

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab15.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 15.1 | Posterior prevalensi stunting dengan rejection dan importance sampling | [`rejection_importance.py`](kode/bab15/rejection_importance.py) |
| 15.2 | Metropolis dari nol untuk regresi logistik stunting–sanitasi | [`metropolis.py`](kode/bab15/metropolis.py) |
| 15.3 | Gibbs sampling untuk rerata dan presisi Gaussian | [`gibbs_gauss.py`](kode/bab15/gibbs_gauss.py) |
| 15.4 | Model hierarkis prevalensi stunting dengan PyMC | [`hierarkis_stunting.py`](kode/bab15/hierarkis_stunting.py) |

### Bab 16. Jaringan Saraf Tiruan

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab16.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 16.1 | Jaringan dua lapis dan backpropagation dengan NumPy | [`mlp_numpy.py`](kode/bab16/mlp_numpy.py) |
| 16.2 | Jaringan yang sama dalam PyTorch, dengan tiga optimizer | [`optimisasi_torch.py`](kode/bab16/optimisasi_torch.py) |
| 16.3 | Weight decay, dropout, dan early stopping pada data latih kecil | [`regularisasi_torch.py`](kode/bab16/regularisasi_torch.py) |
| 16.4 | Mixture density network untuk masalah invers | [`mdn_torch.py`](kode/bab16/mdn_torch.py) |

### Bab 17. Dari Jaringan Saraf ke Deep Learning

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab17.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 17.1 | MLP dan CNN untuk citra daun padi sintetis | [`cnn_daun.py`](kode/bab17/cnn_daun.py) |
| 17.2 | Transformer kecil dari nol untuk sentimen ulasan | [`atensi_ulasan.py`](kode/bab17/atensi_ulasan.py) |
| 17.3 | Kerangka fine-tuning IndoBERT untuk sentimen (tidak dijalankan di buku ini) | [`indobert_sentimen.py`](kode/bab17/indobert_sentimen.py) |
| 17.4 | VAE dengan ruang laten dua dimensi untuk glif Bab 16 | [`vae_glif.py`](kode/bab17/vae_glif.py) |

### Bab 18. Ketidakpastian, Kalibrasi, dan Keputusan

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsnegara/pola-ml-python/blob/main/notebook/bab18.ipynb)

| Listing | Isi | Berkas |
|---|---|---|
| 18.1 | Satu jaringan, MC dropout, dan deep ensemble untuk data curah hujan | [`hujan_ensemble.py`](kode/bab18/hujan_ensemble.py) |
| 18.2 | Kalibrasi pascapelatihan untuk klasifikasi rumah tangga miskin | [`kalibrasi_bansos.py`](kode/bab18/kalibrasi_bansos.py) |
| 18.3 | Deteksi masukan asing dengan ensemble lima jaringan | [`ood_glif.py`](kode/bab18/ood_glif.py) |
| 18.4 | Selang konformal untuk pengeluaran rumah tangga dan keputusan tiga arah | [`konformal_bansos.py`](kode/bab18/konformal_bansos.py) |

Listing 17.3 (IndoBERT) memerlukan `transformers`, `datasets` dan koneksi ke
Hugging Face. Listing itu tidak dijalankan di buku dan tidak dimuat di notebook.

## Lisensi

Kode dan notebook dalam repositori ini dilisensikan dengan
[lisensi MIT](LICENSE). Teks buku tidak termasuk dalam repositori ini dan
dilindungi hak cipta.

## Mengutip

```
Handayani, L., & Insani, F. (2027). Pengenalan Pola dan Machine Learning: Pendekatan
Probabilistik dengan Python. Pekanbaru: Nalar Press Indonesia.
```

Laporan kesalahan pada kode atau buku dapat dikirim lewat *Issues* di
repositori ini.
