import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

# Ganti 'wine_dataset.csv' dengan nama file dataset yang sesuai
# Pastikan file dataset sudah ada di direktori lokal atau gunakan URL yang valid
file_path = "Wholesale customers data.csv"
wc_data = pd.read_csv(file_path)

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
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)

# Latih model pada data pelatihan
et_model.fit(X_train, y_train)

# Fungsi untuk memprediksi kelas 
def predict_wc_channel(features):
    prediction = et_model.predict([features])
    return prediction[0]

# Judul aplikasi
st.title("Aplikasi Untuk Memprediksi Jumlah Channel Yang Digunakan pada Wholesale Customers Datasets") 

# Tampilkan akurasi model
y_pred = et_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Akurasi Model: {accuracy:.2f}")

# Tombol untuk melakukan prediksi
input_features = {}
for feature_name in wc_data.columns[:-1]:  
    value = st.slider(f"Masukkan nilai untuk {feature_name}", 
                      min_value=int(wc_data[feature_name].min()), 
                      max_value=int(wc_data[feature_name].max()))
    input_features[feature_name] = value

if st.button("Prediksi Channel yang digunakan"):
    if len(input_features) == len(wc_data.columns) - 1:  
        prediction = predict_wc_channel(list(input_features.values()))
        st.write(f"Prediksi Channelnya adalah: {prediction}")
    else:
        st.write("Harap masukkan nilai untuk semua fitur")

# Tampilkan dataset jika diinginkan
if st.checkbox("Tampilkan Dataset"):
    st.write(wc_data)
