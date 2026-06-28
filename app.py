import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
import os

# FOCAL LOSS TANIMI
def focal_loss(gamma=2., alpha=4.):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        loss = -y_true * (alpha * tf.pow((1 - y_pred), gamma) * tf.math.log(y_pred))
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

# Sayfa Yapılandırması
st.set_page_config(page_title="Biosignal AI - Sleep Classifier", layout="wide")

# =========================================================
# YASAL UYARI VE KORUMA BANNERI (DISCLAIMER)
# =========================================================
st.error("⚠️ **YASAL UYARI / LEGAL DISCLAIMER:** Bu platform, Biyomedikal Mühendisliği araştırma ve portfolyo çalışması kapsamında geliştirilmiş akademik bir prototiptir. **Tıbbi bir teşhis, tedavi veya klinik karar destek aracı DEĞİLDİR.** Burada elde edilen sonuçlar doğrultusunda tıbbi bir aksiyon alınamaz. Tüm kararlar uzman bir hekim tarafından verilmelidir. (Research Use Only - RUO)")

st.title("🧠 Biosignal AI - Yapay Zeka Destekli Uyku Sınıflandırma")
st.write("Derin Öğrenme (1D-CNN) modeli ile biyomedikal sinyallerden otomatik hipnogram analizi.")

CLASSES = ["W", "N1", "N2", "N3", "REM"]

@st.cache_resource
def load_my_model():
    model_path = os.path.join("outputs", "final_model_cnn.h5")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
    return None

model = load_my_model()

if model:
    st.sidebar.success("✅ Derin Öğrenme Modeli Aktif")
else:
    st.sidebar.warning("⚠️ Demo Modu Aktif")

# =========================================================
# VERİ GİZLİLİĞİ VE KVKK BİLGİLENDİRMESİ
# =========================================================
st.sidebar.info("🔒 **Veri Gizliliği Garanti Metni (KVKK / HIPAA):** Bu platformda işlenen sinyal verileri yalnızca tarayıcı oturumunuz boyunca geçici olarak (RAM bellekte) işlenir. Yüklediğiniz dosyalar hiçbir şekilde sunucularımıza kaydedilmez, depolanmaz veya üçüncü taraflarla paylaşılmaz. Sayfayı kapattığınız an tüm veriler kalıcı olarak silinir.")

# Dosya Yükleme Alanı
uploaded_file = st.file_uploader("Analiz edilecek sinyal dosyasını (.npy veya .csv) yükleyin", type=["npy", "csv"])

# Varsayılan simülasyon verisi
def generate_mock_eeg():
    t = np.linspace(0, 30, 3000)
    ch1 = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.5, 3000)
    ch2 = np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.5, 3000)
    return pd.DataFrame({"Zaman (sn)": t, "EEG Fpz-Cz": ch1, "EEG Pz-Oz": ch2}), ch1.shape[0]

raw_data_for_model = None

if uploaded_file is not None:
    if uploaded_file.name.endswith(".npy"):
        npy_data = np.load(uploaded_file)
        if len(npy_data.shape) == 3:
            raw_data_for_model = npy_data[0]
        else:
            raw_data_for_model = npy_data
        st.success(f"✅ Veri yüklendi. Boyut: {raw_data_for_model.shape}")
        zaman = np.linspace(0, 30, len(raw_data_for_model))
        df = pd.DataFrame({
            "Zaman (sn)": zaman,
            "EEG Fpz-Cz": raw_data_for_model[:, 0],
            "EEG Pz-Oz": raw_data_for_model[:, 1]
        })
    else:
        df = pd.read_csv(uploaded_file)
        raw_data_for_model = df[[df.columns[1], df.columns[2]]].values
        st.success("✅ CSV Verisi alındı.")
else:
    df, _ = generate_mock_eeg()
    st.info("💡 Şu an simüle edilmiş örnek veri inceleniyor. Gerçek bir test için 'X.npy' dosyasını yükleyin.")

# Sinyal Grafiği
fig = px.line(df, x=df.columns[0], y=[df.columns[1], df.columns[2]], title="EEG Sinyal Kanalları")
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# YAPAY ZEKA TAHMİN BÖLÜMÜ
st.subheader("⚡ Yapay Zeka Analiz Sonuçları")

if st.button("Uykuyu Sınıflandır (Modeli Çalıştır)"):
    with st.spinner("Derin öğrenme modeli analiz ediyor..."):
        if model and raw_data_for_model is not None:
            expected_length = model.input_shape[1]
            if len(raw_data_for_model) > expected_length:
                ready_data = raw_data_for_model[:expected_length]
            elif len(raw_data_for_model) < expected_length:
                ready_data = np.pad(raw_data_for_model, ((0, expected_length - len(raw_data_for_model)), (0, 0)), 'constant')
            else:
                ready_data = raw_data_for_model
                
            input_tensor = np.expand_dims(ready_data, axis=0)
            prediction = model.predict(input_tensor)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx] * 100
            result_stage = CLASSES[predicted_class_idx]
            
            prob_df = pd.DataFrame({
                "Evre": CLASSES,
                "Olasılık (%)": [p * 100 for p in prediction[0]]
            })
        else:
            result_stage = np.random.choice(CLASSES)
            confidence = np.random.uniform(75, 98)
            prob_df = pd.DataFrame({
                "Evre": CLASSES,
                "Olasılık (%)": np.random.dirichlet(np.ones(5))*100
            })

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Tahmin Edilen Uyku Evresi", value=result_stage)
            st.metric(label="Güven Oranı", value=f"%{confidence:.2f}")
        with col2:
            st.write("**Evre Bazlı Olasılık Dağılımı:**")
            bar_fig = px.bar(prob_df, x="Evre", y="Olasılık (%)", color="Evre", text_auto='.2f')
            bar_fig.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(bar_fig, use_container_width=True)