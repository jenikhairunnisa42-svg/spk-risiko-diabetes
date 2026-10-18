import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

# Konfigurasi halaman
st.set_page_config(
    page_title="SPK Risiko Diabetes",
    page_icon="🩺",
    layout="wide"
)

st.title("SPK Prediksi Risiko Diabetes")
st.write("Metode K-Nearest Neighbors (KNN)")
st.info("Aplikasi simulasi untuk tugas kuliah.")

# Dataset 10 pasien
data = {
    "Pasien": [
        "P01", "P02", "P03", "P04", "P05",
        "P06", "P07", "P08", "P09", "P10"
    ],
    "Usia": [25, 30, 35, 40, 45, 50, 55, 28, 38, 48],
    "BMI": [21.5, 23.1, 25.0, 27.5, 30.2,
            31.5, 33.0, 22.5, 26.8, 29.5],
    "Glukosa": [90, 95, 105, 120, 140,
                155, 165, 92, 115, 145],
    "Tekanan Darah": [110, 115, 120, 125, 135,
                      140, 145, 112, 122, 130],
    "Riwayat Keluarga": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    "Kelas": [
        "Rendah", "Rendah", "Rendah", "Tinggi", "Tinggi",
        "Tinggi", "Tinggi", "Rendah", "Rendah", "Tinggi"
    ]
}

df = pd.DataFrame(data)

fitur = [
    "Usia", "BMI", "Glukosa",
    "Tekanan Darah", "Riwayat Keluarga"
]

X = df[fitur]
y = df["Kelas"]

# Membuat model KNN
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)

model.fit(X, y)

# Ringkasan data
st.subheader("Ringkasan Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Total Pasien", len(df))
col2.metric("Risiko Rendah", int((y == "Rendah").sum()))
col3.metric("Risiko Tinggi", int((y == "Tinggi").sum()))

# Form input pasien
st.subheader("Input Data Pasien Baru")

with st.form("form_pasien"):

    col1, col2 = st.columns(2)

    with col1:
        usia = st.number_input(
            "Usia (tahun)", 1, 120, 42
        )

        bmi = st.number_input(
            "BMI", 1.0, 100.0, 28.5, step=0.1
        )

        glukosa = st.number_input(
            "Glukosa (mg/dL)", 1, 1000, 130
        )

    with col2:
        tekanan = st.number_input(
            "Tekanan Darah (mmHg)", 1, 300, 128
        )

        riwayat = st.selectbox(
            "Riwayat Diabetes Keluarga",
            [0, 1],
            format_func=lambda x:
                "Tidak Ada (0)" if x == 0 else "Ada (1)"
        )

    tombol = st.form_submit_button(
        "PREDIKSI RISIKO DIABETES",
        use_container_width=True
    )

# Proses prediksi
if tombol:

    pasien_baru = pd.DataFrame(
        [[usia, bmi, glukosa, tekanan, riwayat]],
        columns=fitur
    )

    hasil = model.predict(pasien_baru)[0]

    scaler = model.named_steps["standardscaler"]
    knn = model.named_steps["kneighborsclassifier"]

    X_normal = scaler.transform(X)
    pasien_normal = scaler.transform(pasien_baru)

    jarak, indeks = knn.kneighbors(pasien_normal)

    tetangga = df.iloc[indeks[0]][
        ["Pasien", "Usia", "BMI", "Glukosa",
         "Tekanan Darah", "Kelas"]
    ].copy()

    tetangga["Jarak"] = jarak[0].round(3)

    st.subheader("Hasil Prediksi")

    if hasil == "Tinggi":
        st.error("HASIL MODEL: RISIKO TINGGI")
    else:
        st.success("HASIL MODEL: RISIKO RENDAH")

    st.write("Data pasien yang dimasukkan:")
    st.dataframe(
        pasien_baru,
        hide_index=True,
        use_container_width=True
    )

    st.write("Tiga tetangga terdekat (K = 3):")
    st.dataframe(
        tetangga,
        hide_index=True,
        use_container_width=True
    )

# Menampilkan dataset
st.divider()
st.subheader("Dataset 10 Pasien")

st.dataframe(
    df,
    hide_index=True,
    use_container_width=True
)

st.caption(
    "Riwayat keluarga: 0 = Tidak Ada, 1 = Ada. "
    "Hasil hanya untuk simulasi akademik, bukan diagnosis medis."
)