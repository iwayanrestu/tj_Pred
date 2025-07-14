# app.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model dan data (ganti dengan path file kamu)
model = joblib.load('model_random_forest.pkl')
df_clean = pd.read_csv('data_bersih.csv')  # pastikan sudah bersih dari outlier/NaN

st.title("🚍 Prediksi Jumlah Penumpang Transjakarta")

# Input tanggal
tanggal_input = st.date_input("Pilih Tanggal", value=datetime(2025, 7, 15))
min_tanggal = pd.to_datetime(df_clean['tanggal']).min()
days_since_start = (pd.to_datetime(tanggal_input) - min_tanggal).days
month = tanggal_input.month
day_of_week = tanggal_input.weekday()

# Prediksi
X_input = pd.DataFrame([[days_since_start, month, day_of_week]],
                       columns=['days_since_start', 'month', 'day_of_week'])

if st.button("Prediksi"):
    hasil = model.predict(X_input)[0]
    st.success(f"📊 Perkiraan jumlah penumpang: **{int(hasil):,} orang**")
