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
# Skenario 1
st.write(f"**Skenario 1: Tingkat Penjualan**")
if produk['tingkat_penjualan'] > 300:
    st.write(f"Produk {nama_produk} **Direkomendasikan** berdasarkan tingkat penjualan.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** berdasarkan tingkat penjualan.")

# Skenario 2
st.write(f"**Skenario 2: BPOM**")
if produk['bpom'] == 1:
    st.write(f"Produk {nama_produk} **Direkomendasikan** karena memiliki BPOM.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** karena tidak memiliki BPOM.")

# Skenario 3
st.write(f"**Skenario 3: Rating Shopee**")
if produk['rate'] >= 4.0:
    st.write(f"Produk {nama_produk} **Direkomendasikan** berdasarkan rating Shopee.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** berdasarkan rating Shopee.")
