import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib 
import base64
import os
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from knn import KNN


# Fungsi untuk mengatur latar belakang
def set_background(image_path="bg1.jpg"):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpeg;base64,{encoded_image}');
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.warning("Background image not found. Default background will be used.")

# Atur halaman
st.set_page_config(page_title="Job Recommendation Dashboard", layout="wide")

# Set background
set_background("bg1.jpg")

# Load Dataset and Model
@st.cache_resource
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Dataset tidak ditemukan di path: {file_path}. Pastikan file tersedia.")
        return None

# Load model and transformer objects
try:
    knn_model = joblib.load('knn_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    scaler = joblib.load('normalization_params.joblib')
except FileNotFoundError as e:
    st.error(f"File model atau parameter tidak ditemukan: {e}")
    st.stop()

# Load dataset lama dan dataset baru
data_file_old = "dataset_fiks.csv"
data_file_new = "categorized_jobs.csv"
data_old = load_data(data_file_old)
data_new = load_data(data_file_new)

# Navigation menu
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Menu", ["Tentang Kami", "Tentang Aplikasi", "Rekomendasi Pekerjaan"])

# Judul utama
st.markdown(
    """
    <div style="text-align: center; color: black;">
        <h1 style="font-size: 3em; margin-bottom: 0.2em;">Welcome to the Job Recommendation System</h1>
        <p style="font-size: 1.2em;">Find the best career path tailored to your skills and interests</p>
    </div>
    """,
    unsafe_allow_html=True
)

if menu == "Rekomendasi Pekerjaan":
    if data_old is not None and data_new is not None:
        # Input User
        st.header("Masukkan Data Anda")
        gender = st.selectbox("Jenis Kelamin:", ["Male", "Female", "Prefer not to say"])
        gender_map = {"Male": 0, "Female": 1, "Prefer not to say": 2}
        gender_numeric = gender_map[gender]

        major = st.selectbox("Jurusan Sarjana:", data_old["UG Specialization (Major)"].unique())
        interests = st.text_input("Minat Utama (Pisahkan dengan koma):", placeholder="Contoh: technology, data science, ai").strip()
        skills = st.text_input("Keterampilan (Pisahkan dengan koma):", placeholder="Contoh: python, machine learning, sql").strip()
        cgpa = st.slider("Rata-rata Nilai Akademik:", min_value=2.0, max_value=4.0, step=0.1)

        certification = st.selectbox("Apakah Anda Memiliki Sertifikasi?", ["No", "Yes"])
        certification_map = {"No": 0, "Yes": 1}
        certification_numeric = certification_map[certification]

        certification_course_title = st.text_input("Judul Sertifikasi (jika ada):", placeholder="Contoh: Data Science Certification").strip()

        status = st.selectbox("Status Kerja Saat Ini:", ["Not Working", "Working"])
        status_map = {"Not Working": 0, "Working": 1}
        status_numeric = status_map[status]

        def filter_jobs(data_old, data_new, gender_numeric, major, interests, skills, cgpa, certification_numeric, status_numeric):
            filtered_data_old = data_old[
                (
                    (data_old["Gender"] == gender_numeric) &
                    (data_old["UG Specialization (Major)"] == major) &
                    (data_old["Average CGPA/Percentage"] >= cgpa) &
                    (data_old["Certification Courses"] == certification_numeric) &
                    (data_old["Working Status"] == status_numeric)
                )
            ].copy()

            filtered_data_old["Interest Match"] = filtered_data_old["Interests"].apply(
                lambda x: len(set(x.split(", ")) & set(interests.split(", ")))
            )
            filtered_data_old["Skill Match"] = filtered_data_old["Skills"].apply(
                lambda x: len(set(x.split(", ")) & set(skills.split(", ")))
            )
            filtered_data_old["Total Match"] = filtered_data_old["Interest Match"] + filtered_data_old["Skill Match"]

            recommendations_old = filtered_data_old.sort_values(by="Total Match", ascending=False).head(5)

            recommendations_old["Job Titles"] = recommendations_old["Mapped Category"].apply(
                lambda category: list(data_new[data_new['Category'] == category]['Job Title'].values)
            )

            recommendations_old = recommendations_old.drop_duplicates(subset=["Mapped Category"])

            return recommendations_old

        if st.button("Cari Pekerjaan"):
            if not interests or not skills:
                st.warning("Harap isi minimal satu minat dan satu keterampilan.")
            else:
                recommendations = filter_jobs(data_old, data_new, gender_numeric, major, interests, skills, cgpa, certification_numeric, status_numeric)
                st.subheader("Rekomendasi Pekerjaan:")

                combined_text = interests + ' ' + skills + ' ' + certification_course_title + ' ' + major

                input_tfidf = vectorizer.transform([combined_text]).toarray()

                if input_tfidf.shape[1] < scaler.n_features_in_:
                    padding = np.zeros((1, scaler.n_features_in_ - input_tfidf.shape[1]))
                    input_tfidf = np.hstack((input_tfidf, padding))

                input_normalized = scaler.transform(input_tfidf)
                prediction = knn_model.predict(input_normalized)

                st.write(f"Predicted Career Category: {prediction[0]}")

                if not recommendations.empty:
                    for idx, row in recommendations.iterrows():
                        st.markdown(f"### {row['Mapped Category']}")
                        top_5_jobs = row['Job Titles'][:5]
                        st.write(f"Daftar Job Title: {', '.join(top_5_jobs)}")
                else:
                    st.warning("Tidak ada rekomendasi pekerjaan yang sesuai dengan kriteria Anda.")

elif menu == "Tentang Aplikasi":
    st.title("Tentang Aplikasi Job Recommendation")
    st.markdown(
        """
        ### Deskripsi
        Aplikasi ini membantu pengguna mendapatkan rekomendasi pekerjaan berdasarkan profil mereka, seperti:
        - Jenis kelamin
        - Jurusan sarjana
        - Minat utama
        - Keterampilan
        - Rata-rata nilai akademik
        - Sertifikat
        - Status kerja saat ini
        """
    )

elif menu == "Tentang Kami":
    st.title("Tentang Kami")
    st.markdown(
        """
        ### Mata Kuliah Kecerdasan Artificial
        Dosen Pengampu:
        - Dr. Elly Matul Imah, M.Kom.
        - Yuni Rosita Dewi, S.Si., M.Si
        """
 )
    st.markdown(
        """
        ### Kelompok 6
        Nama Anggota Kelompok:
        1. Layyinatul Qolbiyah (23031554025) 
        2. Thea Bayu Revalina (23031554035) 
        3. Novia Djoend Lestari (23031554220)
        """
 )

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; color: white;">
        <p>&copy; 2025 Job Recommendation System | All Rights Reserved</p>
    </div>
    """,
    unsafe_allow_html=True
)