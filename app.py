import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

st.set_page_config(
    page_title="SPK Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 SPK Prediksi Risiko Diabetes")
st.write("Sistem Pendukung Keputusan menggunakan metode K-Nearest Neighbors (KNN)")

# =====================================
# 1. DATASET PASIEN AWAL
# =====================================
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

fitur = [
    "Usia",
    "BMI",
    "Glukosa",
    "Tekanan Darah",
    "Riwayat Keluarga"
]

# =====================================
# 2. FORM INPUT PASIEN
# =====================================
st.subheader("📝 Input Data Pasien Baru")

with st.form("form_pasien"):
    nama = st.text_input(
        "Nama atau kode pasien",
        placeholder="Contoh: P11"
    )

    kolom1, kolom2 = st.columns(2)

    with kolom1:
        usia = st.number_input(
            "Usia (tahun)", 1, 120, 42
        )
        bmi = st.number_input(
            "BMI", 10.0, 70.0, 28.5, step=0.1
        )
        glukosa = st.number_input(
            "Glukosa (mg/dL)", 20, 600, 130
        )

    with kolom2:
        tekanan = st.number_input(
            "Tekanan Darah (mmHg)", 40, 250, 128
        )
        riwayat = st.selectbox(
            "Riwayat Diabetes Keluarga",
            [0, 1],
            format_func=lambda x:
                "Ada (1)" if x == 1 else "Tidak Ada (0)"
        )

    tombol = st.form_submit_button(
        "🔍 Prediksi Risiko Diabetes"
    )

# Simpan riwayat prediksi selama sesi aktif
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# =====================================
# 3. PERHITUNGAN KNN
# =====================================
if tombol:
    if not nama.strip():
        st.warning("Masukkan nama atau kode pasien terlebih dahulu.")
    else:
        pasien_baru = pd.DataFrame([{
            "Usia": usia,
            "BMI": bmi,
            "Glukosa": glukosa,
            "Tekanan Darah": tekanan,
            "Riwayat Keluarga": riwayat
        }])

        # Standardisasi data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[fitur])
        pasien_scaled = scaler.transform(pasien_baru[fitur])

        # Menghitung jarak Euclidean
        jarak = pairwise_distances(
            pasien_scaled,
            X_scaled,
            metric="euclidean"
        )[0]

        ranking = df[["Pasien", "Risiko"]].copy()
        ranking["Jarak"] = jarak

        ranking = ranking.sort_values(
            "Jarak"
        ).reset_index(drop=True)

        ranking.insert(
            0, "Peringkat",
            range(1, len(ranking) + 1)
        )

        # Mengambil tiga tetangga terdekat
        tetangga = ranking.head(3).copy()

        jumlah_rendah = (
            tetangga["Risiko"] == "Rendah"
        ).sum()

        jumlah_tinggi = (
            tetangga["Risiko"] == "Tinggi"
        ).sum()

        hasil = (
            "Rendah"
            if jumlah_rendah > jumlah_tinggi
            else "Tinggi"
        )

        # =====================================
        # 4. DATA PASIEN BARU
        # =====================================
        st.divider()
        st.subheader("👤 Data Pasien Baru")

        tampilan_pasien = pd.DataFrame({
            "Atribut": [
                "Usia",
                "BMI",
                "Glukosa",
                "Tekanan Darah",
                "Riwayat Keluarga"
            ],
            "Nilai": [
                usia,
                bmi,
                glukosa,
                tekanan,
                "Ada" if riwayat == 1 else "Tidak Ada"
            ]
        })

        st.table(tampilan_pasien)

        # =====================================
        # 5. RANKING JARAK
        # =====================================
        st.subheader("📏 Ranking Jarak (K = 3)")

        ranking_tampil = ranking.copy()
        ranking_tampil["Jarak"] = (
            ranking_tampil["Jarak"].round(4)
        )

        st.dataframe(
            ranking_tampil,
            use_container_width=True,
            hide_index=True
        )

        # =====================================
        # 6. TIGA TETANGGA TERDEKAT
        # =====================================
        st.subheader("🏆 3 Tetangga Terdekat")

        tetangga_tampil = tetangga[
            ["Pasien", "Jarak", "Risiko"]
        ].copy()

        tetangga_tampil["Jarak"] = (
            tetangga_tampil["Jarak"].round(4)
        )

        st.dataframe(
            tetangga_tampil,
            use_container_width=True,
            hide_index=True
        )

        # =====================================
        # 7. PERHITUNGAN SUARA
        # =====================================
        st.subheader("🗳️ Perhitungan Suara Kelas")

        kolom1, kolom2 = st.columns(2)

        kolom1.metric(
            "Risiko Rendah",
            int(jumlah_rendah)
        )

        kolom2.metric(
            "Risiko Tinggi",
            int(jumlah_tinggi)
        )

        st.write(
            f"Risiko Rendah: {jumlah_rendah} suara"
        )
        st.write(
            f"Risiko Tinggi: {jumlah_tinggi} suara"
        )

        # =====================================
        # 8. HASIL PREDIKSI
        # =====================================
        st.subheader("🎯 HASIL PREDIKSI")

        if hasil == "Tinggi":
            st.warning(
                f"Pasien {nama.strip()} — Hasil simulasi: Risiko Tinggi"
            )
        else:
            st.success(
                f"Pasien {nama.strip()} — Hasil simulasi: Risiko Rendah"
            )

        st.caption(
            "Hasil bergantung pada dataset simulasi dan bukan "
            "diagnosis medis."
        )

        # Simpan ke riwayat
        st.session_state.riwayat.append({
            "Pasien": nama.strip(),
            "Usia": usia,
            "BMI": bmi,
            "Glukosa": glukosa,
            "Tekanan Darah": tekanan,
            "Riwayat Keluarga": (
                "Ada" if riwayat == 1 else "Tidak Ada"
            ),
            "Risiko": hasil
        })

# =====================================
# 9. RIWAYAT PREDIKSI
# =====================================
st.divider()
st.subheader("📑 Riwayat Prediksi")

if st.session_state.riwayat:
    riwayat_df = pd.DataFrame(
        st.session_state.riwayat
    )

    st.dataframe(
        riwayat_df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.caption("Belum ada pasien baru yang diprediksi.")
