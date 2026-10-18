import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# =====================================
# PENGATURAN HALAMAN
# =====================================
st.set_page_config(
    page_title="SPK Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 SPK Prediksi Risiko Diabetes")
st.write(
    "Sistem Pendukung Keputusan menggunakan "
    "metode K-Nearest Neighbors (KNN)"
)
st.info(
    "Aplikasi simulasi untuk tugas kuliah, "
    "bukan alat diagnosis medis."
)

# =====================================
# 1. DATASET PASIEN AWAL
# =====================================
data = {
    "Pasien": [
        "P01", "P02", "P03", "P04", "P05",
        "P06", "P07", "P08", "P09", "P10"
    ],
    "Usia": [25, 30, 35, 40, 45, 50, 55, 28, 38, 48],
    "BMI": [
        21.5, 23.1, 25.0, 27.5, 30.2,
        31.5, 33.0, 22.5, 26.8, 29.5
    ],
    "Glukosa": [
        90, 95, 105, 120, 140,
        155, 165, 92, 115, 145
    ],
    "Tekanan Darah": [
        110, 115, 120, 125, 135,
        140, 145, 112, 122, 130
    ],
    "Riwayat Keluarga": [
        0, 0, 0, 1, 1, 1, 1, 0, 0, 1
    ],
    "Risiko": [
        "Rendah", "Rendah", "Rendah", "Tinggi", "Tinggi",
        "Tinggi", "Tinggi", "Rendah", "Rendah", "Tinggi"
    ]
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
# 2. TABEL DATA PASIEN AWAL
# =====================================
st.subheader("📋 Tabel Data Pasien")
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

# =====================================
# 3. RIWAYAT PREDIKSI
# =====================================
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# =====================================
# 4. FORM INPUT PASIEN BARU
# =====================================
st.divider()
st.subheader("📝 Input Data Pasien Baru")

with st.form("form_pasien"):
    nama = st.text_input(
        "Nama atau kode pasien",
        placeholder="Contoh: P11"
    )

    kolom1, kolom2 = st.columns(2)

    with kolom1:
        usia = st.number_input(
            "Usia (tahun)",
            min_value=1,
            max_value=120,
            value=42
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=70.0,
            value=28.5,
            step=0.1
        )

        glukosa = st.number_input(
            "Glukosa (mg/dL)",
            min_value=20,
            max_value=600,
            value=130
        )

    with kolom2:
        tekanan = st.number_input(
            "Tekanan Darah (mmHg)",
            min_value=40,
            max_value=250,
            value=128
        )

        riwayat = st.selectbox(
            "Riwayat Diabetes Keluarga",
            options=[0, 1],
            format_func=lambda x:
                "Ada (1)" if x == 1 else "Tidak Ada (0)"
        )

    tombol = st.form_submit_button(
        "🔍 Prediksi Risiko Diabetes"
    )

# =====================================
# 5. PROSES PERHITUNGAN KNN
# =====================================
if tombol:
    if not nama.strip():
        st.warning(
            "Masukkan nama atau kode pasien terlebih dahulu."
        )

    else:
        pasien_baru = pd.DataFrame([{
            "Usia": usia,
            "BMI": bmi,
            "Glukosa": glukosa,
            "Tekanan Darah": tekanan,
            "Riwayat Keluarga": riwayat
        }])

        # Standardisasi data pasien awal
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[fitur])

        # Standardisasi data pasien baru
        pasien_scaled = scaler.transform(
            pasien_baru[fitur]
        )

        # Menghitung jarak Euclidean
        jarak = pairwise_distances(
            pasien_scaled,
            X_scaled,
            metric="euclidean"
        )[0]

        # Membuat tabel ranking jarak
        ranking = df[["Pasien", "Risiko"]].copy()
        ranking["Jarak"] = jarak

        ranking = ranking.sort_values(
            by="Jarak"
        ).reset_index(drop=True)

        ranking.insert(
            0,
            "Peringkat",
            range(1, len(ranking) + 1)
        )

        # Mengambil 3 tetangga terdekat
        tetangga = ranking.head(3).copy()

        jumlah_rendah = int(
            (tetangga["Risiko"] == "Rendah").sum()
        )

        jumlah_tinggi = int(
            (tetangga["Risiko"] == "Tinggi").sum()
        )

        # Menentukan hasil mayoritas
        if jumlah_rendah > jumlah_tinggi:
            hasil = "Rendah"
        else:
            hasil = "Tinggi"

        # =====================================
        # 6. DATA PASIEN BARU
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
        # 7. RANKING JARAK
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
        # 8. TIGA TETANGGA TERDEKAT
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
        # 9. PERHITUNGAN SUARA KELAS
        # =====================================
        st.subheader("🗳️ Perhitungan Suara Kelas")

        kolom1, kolom2 = st.columns(2)

        kolom1.metric(
            "Risiko Rendah",
            jumlah_rendah
        )

        kolom2.metric(
            "Risiko Tinggi",
            jumlah_tinggi
        )

        st.write(
            f"Risiko Rendah: {jumlah_rendah} suara"
        )

        st.write(
            f"Risiko Tinggi: {jumlah_tinggi} suara"
        )

        # =====================================
        # 10. HASIL PREDIKSI
        # =====================================
        st.subheader("🎯 HASIL PREDIKSI")

        if hasil == "Tinggi":
            st.warning(
                f"Pasien {nama.strip()} — "
                "Hasil simulasi: Risiko Tinggi"
            )
        else:
            st.success(
                f"Pasien {nama.strip()} — "
                "Hasil simulasi: Risiko Rendah"
            )

        st.caption(
            "Hasil ini berdasarkan dataset simulasi, "
            "bukan diagnosis medis."
        )

        # =====================================
        # 11. SIMPAN RIWAYAT
        # =====================================
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
# 12. TABEL RIWAYAT PREDIKSI
# =====================================
st.divider()
st.subheader("📑 Tabel Riwayat Prediksi")

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
    st.caption(
        "Belum ada pasien baru yang diprediksi."
    )
