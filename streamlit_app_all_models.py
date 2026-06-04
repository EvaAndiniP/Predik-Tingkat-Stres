
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Load the scaler
scaler_filename = os.path.join(script_dir, 'deployment_models', 'scaler.pkl')
try:
    scaler = joblib.load(scaler_filename)
except FileNotFoundError:
    st.error(f"Scaler file not found at {scaler_filename}. Make sure 'deployment_models' folder and 'scaler.pkl' are in the same directory as this script.")
    st.stop()


# Load all models
loaded_models = {}
model_paths_map = {
        'Random Forest (Baseline)': os.path.join(script_dir, 'deployment_models', 'Random_Forest_Baseline.pkl'),
        'SVM (Baseline)': os.path.join(script_dir, 'deployment_models', 'SVM_Baseline.pkl'),
        'XGBoost (Baseline)': os.path.join(script_dir, 'deployment_models', 'XGBoost_Baseline.pkl'),
        'Naive Bayes (Baseline)': os.path.join(script_dir, 'deployment_models', 'Naive_Bayes_Baseline.pkl'),
        'Random Forest (SMOTE)': os.path.join(script_dir, 'deployment_models', 'Random_Forest_SMOTE.pkl'),
        'SVM (SMOTE)': os.path.join(script_dir, 'deployment_models', 'SVM_SMOTE.pkl'),
        'XGBoost (SMOTE)': os.path.join(script_dir, 'deployment_models', 'XGBoost_SMOTE.pkl'),
        'Naive Bayes (SMOTE)': os.path.join(script_dir, 'deployment_models', 'Naive_Bayes_SMOTE.pkl'),
        'Random Forest (Tuned)': os.path.join(script_dir, 'deployment_models', 'Random_Forest_Tuned.pkl'),
        'SVM (Tuned)': os.path.join(script_dir, 'deployment_models', 'SVM_Tuned.pkl'),
        'XGBoost (Tuned)': os.path.join(script_dir, 'deployment_models', 'XGBoost_Tuned.pkl'),
        'Naive Bayes (Tuned)': os.path.join(script_dir, 'deployment_models', 'Naive_Bayes_Tuned.pkl')
}

for original_name, model_path in model_paths_map.items():
    try:
        loaded_models[original_name] = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Error loading model '{original_name}' from '{model_path}'. Make sure all model files are in 'deployment_models' folder.")
        st.stop()


st.title('Prediksi Tingkat Stres Siswa')
st.write('Aplikasi ini memprediksi tingkat stres siswa (Rendah, Sedang, Tinggi) berdasarkan faktor akademik dan lingkungan. Anda dapat memilih model yang berbeda untuk membandingkan prediksi.')

# Model selection dropdown
selected_model_name = st.sidebar.selectbox(
    'Pilih Model untuk Prediksi:',
    ['Random Forest (Baseline)', 'SVM (Baseline)', 'XGBoost (Baseline)', 'Naive Bayes (Baseline)', 'Random Forest (SMOTE)', 'SVM (SMOTE)', 'XGBoost (SMOTE)', 'Naive Bayes (SMOTE)', 'Random Forest (Tuned)', 'SVM (Tuned)', 'XGBoost (Tuned)', 'Naive Bayes (Tuned)']
)

model = loaded_models[selected_model_name]

st.sidebar.header('Input Fitur Siswa')

# Feature names (assuming they are always the same and in the same order)
feature_names = ['academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns', 'extracurricular_activities', 'noise_level', 'living_conditions', 'safety', 'basic_needs', 'social_support', 'peer_pressure', 'bullying']

# Create input widgets for each feature in the sidebar
def user_input_features():
    data = {}
    for feature in feature_names:
        # Assuming features range from 0 to 4 based on dataset description
        data[feature] = st.sidebar.slider(feature.replace('_', ' ').title(), 0, 4, 2)
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('Input Pengguna')
st.write(df_input)

# Scale the input features
df_input_scaled = scaler.transform(df_input)

# Make prediction
if st.button('Prediksi Tingkat Stres'):
    prediction = model.predict(df_input_scaled)
    stress_levels = ['Rendah', 'Sedang', 'Tinggi']
    predicted_stress_level = stress_levels[prediction[0]]

    st.subheader('Hasil Prediksi')
    st.success(f'Tingkat Stres yang Diprediksi: **{predicted_stress_level}**')

    if hasattr(model, "predict_proba"):
        prediction_proba = model.predict_proba(df_input_scaled)
        st.subheader('Probabilitas Prediksi')
        proba_df = pd.DataFrame(prediction_proba, columns=stress_levels)
        st.write(proba_df)
    else:
        st.info('Model ini tidak menyediakan probabilitas prediksi.')
