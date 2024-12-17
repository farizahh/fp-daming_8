# Sistem Rekomendasi Produk Sunscreen

Sistem rekomendasi produk sunscreen ini dibangun menggunakan model machine learning untuk memberikan rekomendasi produk berdasarkan beberapa kriteria, seperti tingkat penjualan, BPOM, dan rating di Shopee. Selain itu, sistem ini juga memberikan rekomendasi keseluruhan berdasarkan model pelatihan yang menggunakan algoritma Logistic Regression.

## Deskripsi

Aplikasi ini memungkinkan pengguna untuk memilih produk sunscreen dan mendapatkan rekomendasi apakah produk tersebut layak direkomendasikan berdasarkan beberapa skenario:
- **Skenario 1**: Berdasarkan tingkat penjualan produk.
- **Skenario 2**: Berdasarkan apakah produk memiliki BPOM atau tidak.
- **Skenario 3**: Berdasarkan rating produk di Shopee.

## Teknologi yang Digunakan
- **Python** (3.7+)
- **Streamlit**: Untuk membangun antarmuka pengguna berbasis web.
- **Pandas**: Untuk pengolahan data.
- **Scikit-learn**: Untuk pelatihan dan penggunaan model machine learning.
- **Joblib**: Untuk menyimpan dan memuat model machine learning.

## Dataset
- **dataset.csv**: Dataset asli sebelum dilakukan preprocessing data, termasuk cleaning dan encoding.
- **cleansed_dataset.csv**: Dataset kedua, hasil dari preprocessing data, dengan tambahan atribut SPF dan PA.
- **dataset_dengan_rekomendasi.csv**: Dataset terakhir yang digunakan dalam pelatihan model

