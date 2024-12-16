import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Memuat model yang telah dilatih
with open('rekomendasi_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Memuat scaler yang digunakan dalam pelatihan
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Membaca dataset untuk produk
data = pd.read_csv('dataset_dengan_rekomendasi.csv')

# Menampilkan dropdown untuk memilih produk
nama_produk = st.selectbox("Pilih nama produk:", data['nama_produk'])

# Ambil data produk berdasarkan nama
produk = data[data['nama_produk'] == nama_produk].iloc[0]

# Menampilkan hasil untuk skenario 1: Berdasarkan tingkat penjualan
st.subheader(f"Hasil Rekomendasi Berdasarkan Tingkat Penjualan untuk {nama_produk}")
if produk['tingkat_penjualan'] > 300:
    st.write(f"Produk {nama_produk} **Direkomendasikan** berdasarkan tingkat penjualan.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** berdasarkan tingkat penjualan.")

# Menampilkan hasil untuk skenario 2: Berdasarkan BPOM
st.subheader(f"Hasil Rekomendasi Berdasarkan BPOM untuk {nama_produk}")
if produk['bpom'] == 1:
    st.write(f"Produk {nama_produk} **Direkomendasikan** karena memiliki BPOM.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** karena tidak memiliki BPOM.")

# Menampilkan hasil untuk skenario 3: Berdasarkan rating Shopee
st.subheader(f"Hasil Rekomendasi Berdasarkan Rating Shopee untuk {nama_produk}")
if produk['rate'] >= 4.0:
    st.write(f"Produk {nama_produk} **Direkomendasikan** berdasarkan rating Shopee.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** berdasarkan rating Shopee.")

# Menyiapkan data untuk prediksi
sample_input = pd.DataFrame({
    'tingkat_penjualan': [produk['tingkat_penjualan']],
    'bpom': [produk['bpom']],
    'rate': [produk['rate']],
    'SPF': [produk['SPF']],
    'PA': [produk['PA']]
})

# Melakukan transformasi data dan prediksi
sample_input_scaled = scaler.transform(sample_input)
prediction = model.predict(sample_input_scaled)[0]

# Menampilkan hasil prediksi
if prediction == 1:
    st.write(f"Produk {nama_produk} **Direkomendasikan** secara keseluruhan.")
else:
    st.write(f"Produk {nama_produk} **Tidak Direkomendasikan** secara keseluruhan.")
