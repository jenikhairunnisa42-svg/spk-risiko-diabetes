
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="SPK Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 SPK Prediksi Risiko Diabetes")
st.write("Sistem Pendukung Keputusan menggunakan metode K-Nearest Neighbors (KNN)")
st.info("Aplikasi simulasi untuk tugas kuliah, bukan alat diagnosis medis.")

# Data pasien awal
data = {
    "Pasien": ["P01", "P02", "P03", "P04", "P05",
               "P06", "P07", "P08", "P09", "P10"],
    "Usia": [25, 30, 35, 40, 45, 50, 55, 28, 38, 48],
    "BMI": [21.5, 23.1, 25.0, 27.5, 30.2, 31.5, 33.0, 22.5, 26.8, 29.5],
    "Glukosa": [90, 95, 105, 120, 140, 155, 165, 92, 115, 145],
    "Tekanan Darah": [110, 115, 120, 125, 135, 140, 145, 112, 122, 130],
    "Riwayat Keluarga": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    "Risiko": ["Rendah", "Rendah", "Rendah", "Tinggi", "Tinggi",
               "Tinggi", "Tinggi", "Rendah", "Rendah", "Tinggi"]
}

df = pd.DataFrame(data)

# Membuat model KNN
fitur = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]
X = df[fitur]
y = df["Risiko"]

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)
model.fit(X, y)

# Menyimpan riwayat prediksi selama sesi aplikasi
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# Ringkasan data
st.subheader("📊 Ringkasan Dataset")
kolom1, kolom2, kolom3 = st.columns(3)
kolom1.metric("Total Pasien Awal", len(df))
kolom2.metric("Risiko Rendah", (df["Risiko"] == "Rendah").sum())
kolom3.metric("Risiko Tinggi", (df["Risiko"] == "Tinggi").sum())

# Tabel data awal
st.subheader("📋 Tabel Data Pasien")
st.dataframe(df, use_container_width=True, hide_index=True)

# Input pasien baru
st.subheader("📝 Input Data Pasien Baru")

with st.form("form_pasien"):
    nama = st.text_input("Nama atau kode pasien", placeholder="Contoh: P11")

    kolom1, kolom2 = st.columns(2)

    with kolom1:
        usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=42)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0,
                              value=28.5, step=0.1)
        glukosa = st.number_input("Glukosa (mg/dL)", min_value=20,
                                  max_value=600, value=130)

    with kolom2:
        tekanan = st.number_input("Tekanan Darah (mmHg)",
                                  min_value=40, max_value=250, value=128)
        riwayat = st.selectbox(
            "Riwayat Diabetes Keluarga",
            options=[0, 1],
            format_func=lambda x: "Ada (1)" if x == 1 else "Tidak Ada (0)"
        )

    tombol = st.form_submit_button("🔍 Prediksi Risiko Diabetes")

if tombol:
    if not nama.strip():
        st.warning("Masukkan nama atau kode pasien terlebih dahulu.")
    else:
        input_pasien = pd.DataFrame([{
            "Usia": usia,
            "BMI": bmi,
            "Glukosa": glukosa,
            "Tekanan Darah": tekanan,
            "Riwayat Keluarga": riwayat
        }])

        hasil = model.predict(input_pasien)[0]
        probabilitas = model.predict_proba(input_pasien)[0]
        keyakinan = max(probabilitas) * 100

        pasien_baru = {
            "Pasien": nama.strip(),
            "Usia": usia,
            "BMI": bmi,
            "Glukosa": glukosa,
            "Tekanan Darah": tekanan,
            "Riwayat Keluarga": riwayat,
            "Risiko": hasil
        }

        st.session_state.riwayat.append(pasien_baru)

        st.subheader("📌 Hasil Prediksi Pasien")
        st.write(f"**Kode pasien:** {nama.strip()}")

        if hasil == "Tinggi":
            st.warning("Hasil simulasi KNN: Risiko Tinggi")
        else:
            st.success("Hasil simulasi KNN: Risiko Rendah")

        st.write(f"Skor proporsi suara kelas mayoritas KNN: {keyakinan:.1f}%")
        st.caption(
            "Skor ini bukan probabilitas klinis. Hasil hanya berdasarkan "
            "dataset simulasi yang digunakan."
        )

# Riwayat prediksi
st.subheader("📑 Tabel Riwayat Prediksi")

if st.session_state.riwayat:
    riwayat_df = pd.DataFrame(st.session_state.riwayat)
    st.dataframe(riwayat_df, use_container_width=True, hide_index=True)

    csv = riwayat_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Riwayat Prediksi (CSV)",
        data=csv,
        file_name="riwayat_prediksi_diabetes.csv",
        mime="text/csv"
    )
else:
    st.caption("Belum ada pasien baru yang diprediksi.")
