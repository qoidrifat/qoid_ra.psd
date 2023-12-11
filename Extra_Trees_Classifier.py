import pandas as pd
from sklearn.tree import ExtraTreeClassifier  # Mengganti RandomForestClassifier dengan ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

# Ganti 'wine_dataset.csv' dengan nama file dataset yang sesuai
url = "https://raw.githubusercontent.com/qoidrifat/psd.github.io/master/Wholesale%20customers%20data.csv"
wc_data = pd.read_csv(url)

# Pindahkan kolom 'Channel' ke paling kanan
if 'Channel' in wc_data.columns:
    columns = [col for col in wc_data.columns if col != 'Channel'] + ['Channel']
    wc_data = wc_data[columns]

# Pisahkan atribut dan label
X = wc_data.drop('Channel', axis=1)
y = wc_data['Channel']

# Pisahkan dataset menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model Extra Trees Classifier
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)  # Mengganti RandomForestClassifier dengan ExtraTreesClassifier

# Latih model pada data pelatihan
et_model.fit(X_train, y_train)

# Fungsi untuk memprediksi kelas 
def predict_wc_channel(features):
    prediction = et_model.predict([features])
    return prediction[0]

# Judul aplikasi
st.title("Aplikasi Untuk Mengklasifikasi Jumlah Channel Yang Digunakan pada Wholesale Customers Datasets")  # Mengganti judul aplikasi

# Tampilkan akurasi model
y_pred = et_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Akurasi Model: {accuracy:.2f}")

# Memungkinkan pengguna memasukkan semua fitur
input_features = []
for feature_name in wc_data.columns[:-1]:  # Menghindari kolom 'class'
    value = st.slider(f"Masukkan nilai untuk {feature_name}", min_value=min(wc_data[feature_name]), max_value=max(wc_data[feature_name]))
    input_features.append(value)

# Tombol untuk melakukan prediksi
if st.button("Prediksi Channel yang digunakan "):
    if len(input_features) == len(wc_data.columns) - 1:  # Memastikan ada 13 nilai input
        prediction = predict_wc_channel(input_features)
        st.write(f"Prediksi Channelnya adalah: {prediction}")
    else:
        st.write("Harap masukkan nilai untuk semua fitur")

# Tampilkan dataset jika diinginkan
if st.checkbox("Tampilkan Dataset"):
    st.write(wc_data)
