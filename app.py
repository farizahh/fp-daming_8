import streamlit as st
import pandas as pd
import joblib

# Membaca dataset dari file CSV
data = pd.read_csv('dataset_dengan_rekomendasi.csv')

# Menampilkan nama produk sebagai dropdown
nama_produk = st.selectbox("Pilih nama produk:", data['nama_produk'])

# Ambil data produk berdasarkan nama
produk = data[data['nama_produk'] == nama_produk].iloc[0]

# Load trained model
model = joblib.load('rekomendasi_model.pkl')  # Pastikan path file model sesuai

# Menyiapkan input untuk prediksi
sample_input = [[
    produk['tingkat_penjualan'],
    produk['bpom'],
    produk['rate'],
    produk['SPF'],
    produk['PA']
]]

# Prediksi dari model
prediction = model.predict(sample_input)[0]

# Menampilkan hasil prediksi
st.subheader(f"Hasil Rekomendasi Berdasarkan Model untuk {nama_produk}")
if prediction == 1:
    st.success(f"Produk {nama_produk} **Direkomendasikan**")
else:
    st.error(f"Produk {nama_produk} **Tidak Direkomendasikan**")

# Hasil dari Skenario 1: Berdasarkan tingkat penjualan
st.subheader(f"Hasil Rekomendasi Berdasarkan Tingkat Penjualan untuk {nama_produk}")
if produk['tingkat_penjualan'] > 300:
    st.write(f"Produk {nama_produk} **Direkomendasikan** berdasarkan tingkat penjualan.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** berdasarkan tingkat penjualan.")

# Hasil dari Skenario 2: Berdasarkan BPOM
st.subheader(f"Hasil Rekomendasi Berdasarkan BPOM untuk {nama_produk}")
if produk['bpom'] == 1:
    st.write(f"Produk {nama_produk} **Direkomendasikan** karena memiliki BPOM.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** karena tidak memiliki BPOM.")

# Hasil dari Skenario 3: Berdasarkan rating Shopee
st.subheader(f"Hasil Rekomendasi Berdasarkan Rating Shopee untuk {nama_produk}")
if produk['rate'] >= 4.0:
    st.write(f"Produk {nama_produk} **Direkomendasikan** berdasarkan rating Shopee.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** berdasarkan rating Shopee.")
