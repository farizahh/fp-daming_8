import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan scaler yang sudah disimpan
model = joblib.load('rekomendasi_model.pkl')  # Memuat model yang telah dilatih
scaler = joblib.load('scaler.pkl')  # Memuat scaler yang telah dilatih

# Membaca dataset
data = pd.read_csv('dataset_dengan_rekomendasi.csv')


# Menampilkan selectbox untuk memilih produk
nama_produk = st.selectbox("Pilih nama produk:", data['nama_produk'])

# Ambil data produk berdasarkan nama
produk = data[data['nama_produk'] == nama_produk].iloc[0]

# Menampilkan detail produk
st.write(f"Menampilkan data untuk produk: {nama_produk}")
st.write(f"Tingkat Penjualan: {produk['tingkat_penjualan']}")
st.write(f"BPOM: {produk['bpom']}")
st.write(f"Rating Shopee: {produk['rate']}")

# Persiapkan data input untuk prediksi
input_data = np.array([produk['tingkat_penjualan'], produk['bpom'], produk['rate']]).reshape(1, -1)

# Lakukan scaling pada data input
input_data_scaled = scaler.transform(input_data)

# Melakukan prediksi dengan model
prediction = model.predict(input_data_scaled)[0]

# Tampilkan hasil prediksi berdasarkan skenario
st.subheader(f"Hasil Rekomendasi untuk {nama_produk}")

# Hasil dari Skenario 1: Berdasarkan tingkat penjualan
st.subheader(f"Skenario 1: Berdasarkan Tingkat Penjualan")
if produk['tingkat_penjualan'] > 300:
    st.write(f"Produk {nama_produk} **Direkomendasikan** berdasarkan tingkat penjualan.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** berdasarkan tingkat penjualan.")

# Hasil dari Skenario 2: Berdasarkan BPOM
st.subheader(f"Skenario 2: Berdasarkan BPOM")
if produk['bpom'] == 1:
    st.write(f"Produk {nama_produk} **Direkomendasikan** karena memiliki BPOM.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** karena tidak memiliki BPOM.")

# Hasil dari Skenario 3: Berdasarkan rating Shopee
st.subheader(f"Skenario 3: Berdasarkan Rating Shopee")
if produk['rate'] >= 4.0:
    st.write(f"Produk {nama_produk} **Direkomendasikan** berdasarkan rating Shopee.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** berdasarkan rating Shopee.")
