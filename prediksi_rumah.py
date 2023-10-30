import streamlit as st
import pickle

model = pickle.load(open('prediksi_rumah.sav', 'rb'))

st.title('Housing Price Prediction App')

st.write('Please enter the following details for the prediction')

area = st.number_input('Input Luas Tanah')
bedrooms = st.number_input('Jumlah Kamar Tidur')
bathrooms = st.number_input('Jumlah Kamar Mandi')
stories = st.number_input('Jumlah Lantai')

mainroad = st.radio('Jalan Utama', ['Yes', 'No'])
if mainroad == 'Yes':
  mainroad = 0
else:
  mainroad = 1

guestroom = st.radio('Kamar Tamu', ['Yes', 'No'])
if guestroom == 'Yes':
  guestroom = 1
else:
  guestroom = 0

basement = st.radio('Basement', ['Yes', 'No'])
if basement == 'Yes':
  basement = 1
else:
  basement = 0

hotwaterheating = st.radio('Pemanas Air', ['Yes', 'No'])
if hotwaterheating == 'Yes':
  hotwaterheating = 1
else:
  hotwaterheating = 0

airconditioning = st.radio('AC', ['Yes', 'No'])
if airconditioning == 'Yes':
  airconditioning = 0
else:
  airconditioning = 1

parking = st.number_input('Tempat parkir')

prefarea = st.radio('Area Pilihan', ['Yes', 'No'])
if prefarea == 'Yes':
  prefarea = 0
else:
  prefarea = 1

furnishingstatus = st.radio('Status Perabotan', ['furnished', 'semi-furnished', 'unfurnished'])
if furnishingstatus == 'furnished':
  furnishingstatus = 0
elif furnishingstatus == 'semi-furnished':
  furnishingstatus = 1
else:
  furnishingstatus = 2

predict =''

if st.button('Prediksi Harga'):
  predict = model.predict(
    [[area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus]]
  )
  st.write('Estimasi harga Rumah dalam USD:', predict)
  st.write('Estimasi harga Rumah dalam IDR (Juta):', predict * 19000)