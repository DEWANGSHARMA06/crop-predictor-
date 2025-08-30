import streamlit as st
import pandas as pd
import joblib

# Load model and label encoder saved from Colab/VS Code
model = joblib.load('crop_recommender.pkl')
le = joblib.load('label_encoder.pkl')

st.title('Crop Predictor')

# Collect input features from user
N = st.number_input('Nitrogen (N)', value=80.0, min_value=0.0)
P = st.number_input('Phosphorus (P)', value=40.0, min_value=0.0)
K = st.number_input('Potassium (K)', value=38.0, min_value=0.0)
temperature = st.number_input('Temperature (°C)', value=22.0, min_value=0.0)
humidity = st.number_input('Humidity (%)', value=82.0, min_value=0.0)
ph = st.number_input('pH level', value=6.7, min_value=0.0, max_value=14.0)
rainfall = st.number_input('Rainfall (mm)', value=210.0, min_value=0.0)
region = st.text_input('Region (optional)')

# When button clicked, predict crop
if st.button('Predict Crop'):
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    pred_encoded = model.predict(input_df)
    pred_crop = le.inverse_transform(pred_encoded)[0]
    
    st.success(f'Predicted Crop: {pred_crop}')
