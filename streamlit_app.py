import subprocess
import sys
# Cek dan install statsmodels jika belum ada
try:
    import statsmodels
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
    import statsmodels
    
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import os

# ===========================
# 🎨 CSS untuk tampilan
# ===========================
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    .clinic-name {
        font-size: 30px;
        color: #1D4ED8;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================
# 🏥 Judul Aplikasi
# ===========================
st.markdown('<h1 class="title">Prediksi Jumlah Pasien dan Optimalisasi Pengadaan Tenaga Medis Dokter Pada <span>Klinik Rahmani</span></h1>', unsafe_allow_html=True)

# ===========================
# 🩺 Pilih Jenis Dokter
# ===========================
tipe_dokter = st.selectbox('Pilih Jenis Dokter:', ['🩺 Dokter Umum', '🦷 Dokter Gigi'])

# 📅 Pilih Bulan
bulan_list = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
              'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
bulan_pilih = st.selectbox('Pilih Bulan:', bulan_list)

# 🔢 Mapping bulan ke angka
bulan_dict = {bulan: i+1 for i, bulan in enumerate(bulan_list)}
bulan_angka = bulan_dict[bulan_pilih]

# ===========================
# 📥 Load Model & Data Eksternal
# ===========================
try:
    if 'Dokter Umum' in tipe_dokter:
        model_path = 'model.sav'
        data_path = 'data_dokter_umum2024.csv'
        time_per_patient = 15  # Menit per pasien
        jenis_dokter_text = 'Dokter Umum'
        stp_time_month = 110  # Jam kerja STP bulanan
        
    else:
        model_path = 'model_gigi.sav'
        data_path = 'data_dokter_gigi2024.csv'
        time_per_patient = 20  # Menit per pasien
        jenis_dokter_text = 'Dokter Gigi'
        stp_time_month = 120  # Jam kerja STP bulanan

    # Cek apakah model tersedia
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model {model_path} tidak ditemukan!")

    # Load model prediksi
    model = pickle.load(open(model_path, 'rb'))

    # Cek apakah data tersedia
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File data {data_path} tidak ditemukan!")

    # Membaca data pasien 2024
    data_2024 = pd.read_csv(data_path)
    data_2024['bulan_tahun'] = pd.to_datetime(data_2024['bulan_tahun'], format='%Y-%m')
    monthly_visits_2024 = data_2024.groupby('bulan_tahun')['jumlah_kunjungan_per_bulan'].sum().reset_index()

except Exception as e:
    st.error(f"Error saat memuat model atau data: {e}")
    st.stop()  # Hentikan eksekusi jika ada error

# ===========================
# 🔍 Menampilkan Hasil Prediksi
# ===========================
prediksi_bulan = model.predict(start=bulan_angka, end=bulan_angka)
prediksi_pasien = int(prediksi_bulan.iloc[0])
st.write(f"📊 Prediksi Pasien ({jenis_dokter_text}) untuk {bulan_pilih}: {prediksi_pasien} pasien")

# Membandingkan dengan Data Nyata 2024
if bulan_angka <= len(monthly_visits_2024):
    jumlah_pasien_2024 = monthly_visits_2024.iloc[bulan_angka-1]['jumlah_kunjungan_per_bulan']
    perbandingan = prediksi_pasien - jumlah_pasien_2024
    st.write(f"📊 Jumlah Pasien 2024 untuk {bulan_pilih}: {jumlah_pasien_2024} pasien")
    st.write(f"📉 Perbandingan Prediksi dan Realitas: {perbandingan} pasien")

# ===========================
# 🔢 Perhitungan Kebutuhan SDM
# ===========================
wkt_per_month = 9590.00  # Waktu kerja total per bulan (menit)
bkt = prediksi_pasien * time_per_patient  # Beban kerja total (menit)
ftp = (stp_time_month / wkt_per_month) * 100  # Faktor penunjang
stp = 1 / (1 - (ftp / 100))  # Standar tugas penunjang
final_sdm = np.ceil(bkt / (wkt_per_month * stp))  # Estimasi kebutuhan SDM
wkt_optimal_dokter = ((wkt_per_month * stp) / 4) / 60  # Waktu optimal per minggu (jam)

st.write(f"👩‍⚕ Kebutuhan SDM ({jenis_dokter_text}): {final_sdm:.0f} dokter")
st.write(f"⏳ Waktu Optimal per Minggu: {wkt_optimal_dokter:.0f} jam")

# ===========================
# 🔢 Input Manual untuk Prediksi Pasien
# ===========================
st.subheader(f'📥 Masukkan Data Pasien (Manual) - {jenis_dokter_text}')
manual_patients = st.number_input(f'Jumlah Pasien ({jenis_dokter_text}):', min_value=1, step=1)

if st.button(f'🚀 Prediksi Manual ({jenis_dokter_text})'):
    bkt_manual = manual_patients * time_per_patient
    final_sdm_manual = np.ceil(bkt_manual / (wkt_per_month * stp))
    wkt_optimal_dokter_manual = ((wkt_per_month * stp) / 4) / 60

    st.write(f"👨‍⚕ Kebutuhan SDM Manual ({jenis_dokter_text}): {final_sdm_manual:.0f} dokter")
    st.write(f"⏳ Waktu Optimal per Minggu (Manual): {wkt_optimal_dokter_manual:.0f} jam")
