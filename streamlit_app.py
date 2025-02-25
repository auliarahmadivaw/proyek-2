import numpy as np
import pandas as pd
import streamlit as st
import pickle
import plotly.express as px

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
st.markdown('<h1 class="title">Prediksi Jumlah Pasien dan Optimalisasi Kebutuhan Tenaga Medis Dokter Pada Klinik Rahmani</span></h1>', unsafe_allow_html=True)

# ===========================
# 🩺 Pilih Jenis Dokter
# ===========================
tipe_dokter = st.selectbox('Pilih Jenis Dokter:', ['🩺 Dokter Umum', '🦷 Dokter Gigi'])

# ===========================
# 👥 Load Model & Data Eksternal
# ===========================
try:
    if tipe_dokter == '🩺 Dokter Umum':
        model_file = 'model.sav'
        exog ='exog.sav'
        data_file = 'data_dokter_umum2024.csv'
        time_per_patient = 15
        jenis_dokter_text = 'Dokter Umum'
        stp_time_month = 110

        # Membaca data pasien 2024
        data_2024_umum = pd.read_csv('data_dokter_umum2024.csv')
        data_2024_umum['bulan_tahun'] = pd.to_datetime(data_2024_umum['bulan_tahun'], format='%Y-%m')
        monthly_visits_2024_umum = data_2024_umum.groupby('bulan_tahun')['jumlah_kunjungan_per_bulan'].sum().reset_index()

    
    else:  # tipe_dokter == '🦷 Dokter Gigi'
        model_file = 'model_gigi.sav'
        data_file = 'data_dokter_gigi2024.csv'
        time_per_patient = 20
        jenis_dokter_text = 'Dokter Gigi'
        stp_time_month = 120

        # Membaca data pasien 2024
        data_2024_gigi = pd.read_csv('data_dokter_gigi2024.csv')
        data_2024_gigi['bulan_tahun'] = pd.to_datetime(data_2024_gigi['bulan_tahun'], format='%Y-%m')
        monthly_visits_2024_gigi = data_2024_gigi.groupby('bulan_tahun')['jumlah_kunjungan_per_bulan'].sum().reset_index()

    # Load Model
    model = pickle.load(open(model_file, 'rb'))

    # Load Data Pasien
    data_2024 = pd.read_csv(data_file)
    data_2024['bulan_tahun'] = pd.to_datetime(data_2024['bulan_tahun'], format='%Y-%m')
    monthly_visits_2024 = data_2024.groupby('bulan_tahun')['jumlah_kunjungan_per_bulan'].sum().reset_index()
  
except Exception as e:
    st.error(f"⚠ Error saat memuat model atau data: {e}")
    model = None

# ===========================
# 🔍 Menampilkan Hasil Prediksi & Visualisasi
# ===========================
if model:
    prediksi_bulan = model.predict(start=1, end=12).astype(int)
    prediksi_pasien = int(prediksi_bulan.iloc[0])

    # Data perbandingan 2024 vs 2025
    df_prediksi = pd.DataFrame({
        'Bulan': ["Januari", "Februari", "Maret", "April", "Mei", "Juni", 
                  "Juli", "Agustus", "September", "Oktober", "November", "Desember"],
        'Jumlah Pasien 2024': monthly_visits_2024['jumlah_kunjungan_per_bulan'].tolist(),
        'Prediksi 2025': prediksi_bulan.tolist()
    })
    
    # Visualisasi tren pasien
    fig = px.line(df_prediksi, x='Bulan', y=['Jumlah Pasien 2024', 'Prediksi 2025'], markers=True,
                  title=f'📊 Tren Pasien Tahun 2024 dan Prediksi 2025 ({jenis_dokter_text})')
    st.plotly_chart(fig)
    
    # ===========================
    # ⚙ Perhitungan Kebutuhan SDM
    # ===========================
    wkt_per_month = 9590.00  # Waktu kerja total dalam satu bulan
    bkt = prediksi_pasien * time_per_patient  # Beban kerja total
    ftp = (stp_time_month / wkt_per_month) * 100
    stp = 1 / (1 - (ftp / 100))
    final_sdm = np.ceil(bkt / (wkt_per_month * stp))
    wkt_optimal_dokter = ((wkt_per_month * stp) / 4) / 60
    
    st.subheader(f"📌 Kebutuhan SDM {jenis_dokter_text}")
    st.write(f"👨‍⚕ Jumlah Dokter yang Dibutuhkan: {final_sdm:.0f} dokter")
    st.write(f"⏳ Waktu Optimal per Minggu: {wkt_optimal_dokter:.0f} jam")
    
    # ===========================
    # 🔢 Input Manual untuk Prediksi Pasien
    # ===========================
    st.subheader(f'📝 Masukkan Data Pasien (Manual) - {jenis_dokter_text}')
    manual_patients = st.number_input(f'Jumlah Pasien ({jenis_dokter_text}):', min_value=1, step=1)
    
    if st.button(f'🚀 Prediksi Manual ({jenis_dokter_text})'):
        bkt_manual = manual_patients * time_per_patient
        final_sdm_manual = np.ceil(bkt_manual / (wkt_per_month * stp))
        wkt_optimal_dokter_manual = ((wkt_per_month * stp) / 4) / 60
        
        st.write(f"👨‍⚕ Jumlah Dokter (Manual): {final_sdm_manual:.0f} dokter")
        st.write(f"⏳ Waktu Optimal per Minggu (Manual): {wkt_optimal_dokter_manual:.0f} jam")

else:
    st.warning("⚠ Data prediksi tidak ditemukan. Pastikan model sudah dimuat dengan benar.")
