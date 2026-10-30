import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Prediksi Harga Laptop", layout="wide")

# ========== Sidebar dengan Navigasi ==========
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Menu", ["Beranda", "Prediksi Harga", "Tentang Aplikasi", "Kontak / Feedback"])

# ========== Cache untuk Model ==========
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data["model"], data["scaler"]

pipe, scaler = load_model()

# ========== Halaman BERANDA ==========
if page == "Beranda":
    st.title("🎮 Aplikasi Prediksi Harga Laptop")
    
    st.markdown("""
    <div style='text-align: justify; font-size: 18px;'>
        Selamat datang di aplikasi <b>Prediksi Harga Laptop</b>, sebuah sistem berbasis Machine Learning
        yang membantu Anda memperkirakan harga laptop berdasarkan spesifikasi teknis pilihan Anda.
    </div>
    """, unsafe_allow_html=True)

    st.image("laptop_hero.jpg", caption="Ilustrasi Laptop Gaming", use_container_width=True)

    st.markdown("---")

    fitur1, fitur2, fitur3 = st.columns(3)

    with fitur1:
        st.markdown("### ⚙️ Spesifikasi Lengkap")
        st.markdown("Pilih berbagai parameter seperti RAM, SSD, GPU, dan Processor.")

    with fitur2:
        st.markdown("### 🤖 Model Cerdas")
        st.markdown("Model machine learning dilatih dari data real laptop gaming.")

    with fitur3:
        st.markdown("### 📊 Hasil Instan")
        st.markdown("Hasil prediksi langsung ditampilkan secara real-time.")

    st.markdown("---")
    st.info("💡 Gunakan menu di samping kiri untuk berpindah antar halaman.")

# ========== Halaman PREDIKSI HARGA ==========
elif page == "Prediksi Harga":
    st.title("🔍 Prediksi Harga Laptop")

    # Input form dalam satu kolom
    brand = st.selectbox("Brand", ['ASUS', 'ACER', 'MSI', 'HP', 'LENOVO', 'GIGABYTE', 'DELL', 'RAZER'])
    ram = st.selectbox("RAM (GB)", [4, 8, 16, 24, 32, 64])
    processor = st.selectbox("Processor", [
        'Intel i3', 'Intel i5', 'Intel i7', 'Intel i9',
        'AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9'
    ])
    ssd = st.number_input("Kapasitas SSD (GB)", min_value=128, max_value=4096, value=512, step=128)
    ukuran_layar = st.selectbox("Ukuran Layar (Inch)", [13.3, 14.0, 15.6, 16.0, 17.0, 18.0])
    sistem_operasi = st.selectbox("Sistem Operasi", ['Windows 10', 'Windows 11', 'Linux', 'macOS'])
    gpu_brand = st.selectbox("GPU Brand", ['NVIDIA', 'AMD', 'Intel'])
    gpu_model = st.selectbox("GPU Model", [
        'RTX 3050', 'RTX 3050TI', 'RTX 3060', 'RTX 3060TI', 'RTX 3070', 'RTX 3070TI',
        'RTX 3080', 'RTX 3080TI', 'RTX 4050', 'RTX 4050TI', 'RTX 4060', 'RTX 4060TI',
        'RTX 4070', 'RTX 4070TI', 'RTX 4080', 'RTX 4080TI', 'RTX 4090', 'RTX 4090TI',
        'Radeon RX 6600M'
    ])
    vram = st.selectbox("VRAM GPU (GB)", [0, 4, 6, 8, 16, 24, 32, 64])
    gen_processor = st.selectbox("Generasi Processor", list(range(8, 15)))
    seri_processor = st.selectbox("Seri Processor", ['H', 'P', 'U', 'HX', 'HS', 'G'])

    if st.button("Submit"):
        input_df = pd.DataFrame([{
            'Processor': processor,
            'SSD(GB)': ssd,
            'Ukuran Layar': ukuran_layar,
            'RAM(GB)': ram,
            'Sistem Operasi': sistem_operasi,
            'Brand': brand,
            'GPU_brand': gpu_brand,
            'GPU_model': gpu_model,
            'GPU_vram': vram,
            'Gen_processor': gen_processor,
            'Seri_processor': seri_processor
        }])

        try:
            pred_scaled = pipe.predict(input_df)[0]
            pred_rupiah = scaler.inverse_transform([[pred_scaled]])[0][0]

            st.success(f"💰 Perkiraan Harga Laptop: Rp {pred_rupiah:,.0f}")
            st.subheader("📊 Detail Input")
            st.write(input_df)
            st.caption(f"Hasil Prediksi (Skala): {pred_scaled:.4f}")

        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")

# ========== Halaman TENTANG ==========
elif page == "Tentang Aplikasi":
    st.title("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dikembangkan menggunakan:

    - **Python** sebagai bahasa pemrograman utama  
    - **Streamlit** untuk antarmuka web interaktif  
    - **Scikit-learn** untuk pembuatan model machine learning  
    - **Pickle** untuk menyimpan pipeline model dan scaler  
    - **Pandas** untuk manipulasi data  

    #### Cara Kerja:
    - Input pengguna dimasukkan dalam bentuk DataFrame  
    - Model melakukan prediksi dalam skala terstandarisasi  
    - Hasil dikembalikan dalam bentuk harga (Rupiah) setelah transformasi balik dengan `MinMaxScaler`

    #### Sumber Data:
    - Dataset disusun dari spesifikasi laptop gaming berbagai merek dan harga dari marketplace di Indonesia.
    """)

# ========== Halaman KONTAK ==========
elif page == "Kontak / Feedback":
    st.title("📬 Kontak & Feedback")
    st.markdown("""
    Anda dapat memberikan saran, kritik, atau bertanya lebih lanjut melalui:

    - GitHub: [github.com/farismns](https://github.com/farismns)
    - Email: [faris.m.saputra@gmail.com](mailto:faris.m.saputra@gmail.com)

    Atau isi formulir berikut:
    """)

    name = st.text_input("Nama")
    email = st.text_input("Email")
    message = st.text_area("Pesan atau Saran")

    if st.button("Kirim Pesan"):
        if name and email and message:
            st.success("✅ Terima kasih atas pesan Anda! Kami akan segera menindaklanjuti.")
        else:
            st.warning("⚠️ Harap lengkapi semua kolom terlebih dahulu.")
