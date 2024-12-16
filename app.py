import streamlit as st
import pandas as pd
import joblib

# Load dataset
data = pd.read_csv('dataset_dengan_rekomendasi.csv')

# Load trained model
model = joblib.load('rekomendasi_model.pkl')

# Aplikasi Streamlit
st.title("Rekomendasi Produk Sunscreen")

# Dropdown untuk memilih produk
nama_produk = st.selectbox("Pilih nama produk:", data['nama_produk'].unique())

# Ambil data produk berdasarkan nama
produk = data[data['nama_produk'] == nama_produk].iloc[0]

# Tampilkan informasi produk
st.subheader(f"Informasi Produk: {nama_produk}")
st.write(f"Tingkat Penjualan: {produk['tingkat_penjualan']}")
st.write(f"BPOM: {'Ada' if produk['bpom'] == 1 else 'Tidak Ada'}")
st.write(f"Rating Shopee: {produk['rate']}")
st.write(f"SPF: {produk['SPF']}")
st.write(f"PA: {produk['PA']}")

# Input untuk prediksi
st.subheader("Hasil Rekomendasi:")
sample_input = [[
    produk['tingkat_penjualan'], 
    produk['bpom'], 
    produk['rate'], 
    produk['SPF'], 
    produk['PA']
]]

# Prediksi dari model
prediction = model.predict(sample_input)[0]

if prediction == 1:
    st.success(f"Produk {nama_produk} **Direkomendasikan**")
else:
    st.error(f"Produk {nama_produk} **Tidak Direkomendasikan**")

# Informasi tambahan untuk masing-masing skenario
st.subheader("Detail Skenario:")
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
