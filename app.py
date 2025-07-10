import streamlit as st
import numpy as np
import joblib

st.title("Prediksi Deteksi Ranjau Darat (Land Mines UCI - GaussianNB)")

# Load model & nama fitur
model = joblib.load("model_gaussian_nb.pkl")  
feature_names = joblib.load("feature_names.pkl")

st.markdown("""
Masukkan nilai-nilai sensor untuk memprediksi apakah suatu area mengandung **ranjau darat (Mine)** atau bukan (**Non-Mine**).
""")

# Form input user
with st.form("form_prediksi"):
    st.subheader("Masukkan Nilai Fitur Sensor")

    input_values = []
    for feature in feature_names:
        val = st.number_input(f"{feature}", min_value=-1000.0, max_value=1000.0, value=0.0, step=0.1, format="%.4f")
        input_values.append(val)

    submitted = st.form_submit_button("Prediksi")

# Ketika tombol submit ditekan
if submitted:
    input_array = np.array([input_values])
    pred = model.predict(input_array)[0]

    label = "Mine (Ranjau)" if pred == 1 else "Non-Mine"
    st.success(f"Hasil Prediksi: **{label}**")

# Tombol download model
with open("model_gaussian_nb.pkl", "rb") as f:
    st.download_button("📥 Download Model", f, file_name="model_gaussian_nb.pkl")
