import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# ─── Load Model and Scaler ────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/xgb_tuned_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    threshold = joblib.load('models/best_threshold.pkl')
    return model, scaler, threshold

model, scaler, threshold = load_artifacts()

# ─── Load Feature Names ───────────────────────────────────────
@st.cache_data
def load_feature_names():
    X_train = pd.read_csv('data/processed/X_train.csv')
    return X_train.columns.tolist()

feature_names = load_feature_names()

# ─── Title ────────────────────────────────────────────────────
st.title("🏥 Hospital Readmission Risk Predictor")
st.markdown("### Predict if a diabetic patient will be readmitted within 30 days")
st.markdown("---")

# ─── Sidebar — Patient Information Input ──────────────────────
st.sidebar.header("📋 Patient Information")

age = st.sidebar.selectbox("Age Group", options=[
    '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
])

time_in_hospital = st.sidebar.slider(
    "Time in Hospital (days)", min_value=1, max_value=14, value=4)

num_lab_procedures = st.sidebar.slider(
    "Number of Lab Procedures", min_value=1, max_value=132, value=40)

num_procedures = st.sidebar.slider(
    "Number of Procedures", min_value=0, max_value=6, value=1)

num_medications = st.sidebar.slider(
    "Number of Medications", min_value=1, max_value=81, value=15)

number_outpatient = st.sidebar.slider(
    "Outpatient Visits (past year)", min_value=0, max_value=42, value=0)

number_emergency = st.sidebar.slider(
    "Emergency Visits (past year)", min_value=0, max_value=76, value=0)

number_inpatient = st.sidebar.slider(
    "Inpatient Visits (past year)", min_value=0, max_value=21, value=0)

number_diagnoses = st.sidebar.slider(
    "Number of Diagnoses", min_value=1, max_value=16, value=7)

# Categorical inputs
race = st.sidebar.selectbox("Race", options=[
    'Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'])

gender = st.sidebar.selectbox("Gender", options=['Male', 'Female'])

insulin = st.sidebar.selectbox("Insulin", options=[
    'No', 'Steady', 'Up', 'Down'])

diabetes_med = st.sidebar.selectbox(
    "On Diabetes Medication?", options=['Yes', 'No'])

change = st.sidebar.selectbox(
    "Medication Change During Stay?", options=['Ch', 'No'])

diag_1 = st.sidebar.selectbox("Primary Diagnosis", options=[
    'Circulatory', 'Diabetes_Metabolic', 'Respiratory',
    'Digestive', 'Symptoms', 'Injury_Poisoning',
    'Genitourinary', 'Musculoskeletal', 'Cancer',
    'Infectious', 'Skin_Disease', 'Mental_Health',
    'External_Supplementary', 'Nervous_System',
    'Blood_Disease', 'Pregnancy', 'Congenital', 'Other'
])

max_glu_serum = st.sidebar.selectbox("Max Glucose Serum", options=[
    'None', 'Norm', '>200', '>300'])

A1Cresult = st.sidebar.selectbox("A1C Result", options=[
    'None', 'Norm', '>7', '>8'])

# ─── Feature Engineering ──────────────────────────────────────
def build_input(feature_names):
    """
    Builds input dataframe matching exact training features.
    """
    # Age encoding
    age_map = {
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2,
        '[30-40)': 3, '[40-50)': 4, '[50-60)': 5,
        '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
    }

    # Start with all zeros
    input_dict = {feat: 0 for feat in feature_names}

    # Fill numerical features
    input_dict['age'] = age_map[age]
    input_dict['time_in_hospital'] = time_in_hospital
    input_dict['num_lab_procedures'] = num_lab_procedures
    input_dict['num_procedures'] = num_procedures
    input_dict['num_medications'] = num_medications
    input_dict['number_outpatient'] = number_outpatient
    input_dict['number_emergency'] = number_emergency
    input_dict['number_inpatient'] = number_inpatient
    input_dict['number_diagnoses'] = number_diagnoses

    # Engineered feature
    input_dict['service_utilization'] = (
        number_outpatient + number_emergency + number_inpatient)

    # One hot encoded features — set matching column to 1
    race_col = f'race_{race}'
    if race_col in input_dict:
        input_dict[race_col] = 1

    if gender == 'Male':
        input_dict['gender_Male'] = 1

    insulin_col = f'insulin_{insulin}'
    if insulin_col in input_dict:
        input_dict[insulin_col] = 1

    if diabetes_med == 'Yes':
        input_dict['diabetesMed_Yes'] = 1

    change_col = f'change_{change}'
    if change_col in input_dict:
        input_dict[change_col] = 1

    diag_col = f'diag_1_{diag_1}'
    if diag_col in input_dict:
        input_dict[diag_col] = 1

    glu_col = f'max_glu_serum_{max_glu_serum}'
    if glu_col in input_dict:
        input_dict[glu_col] = 1

    a1c_col = f'A1Cresult_{A1Cresult}'
    if a1c_col in input_dict:
        input_dict[a1c_col] = 1

    return pd.DataFrame([input_dict])

# ─── Prediction ───────────────────────────────────────────────
st.markdown("## 🔍 Prediction Results")

if st.button("🚀 Predict Readmission Risk", use_container_width=True):
    # Build input
    input_df = build_input(feature_names)

    # Scale numerical features
    numerical_cols = [
        'age', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient',
        'number_emergency', 'number_inpatient', 'number_diagnoses',
        'service_utilization'
    ]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Predict
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= threshold)

    # ─── Display Results ──────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("## 🔴 HIGH RISK")
            st.error("Patient likely to be readmitted within 30 days")
        else:
            st.success("## 🟢 LOW RISK")
            st.success("Patient unlikely to be readmitted within 30 days")

    with col2:
        st.metric(
            label="Readmission Probability",
            value=f"{probability:.1%}"
        )

    with col3:
        st.metric(
            label="Decision Threshold",
            value=f"{threshold:.2f}"
        )

    # ─── Risk Gauge ───────────────────────────────────────────
    st.markdown("### 📊 Risk Level")
    st.progress(float(probability))
    
    if probability < 0.3:
        st.info("🟢 Low Risk Zone (< 30%)")
    elif probability < 0.5:
        st.warning("🟡 Moderate Risk Zone (30-50%)")
    else:
        st.error("🔴 High Risk Zone (> 50%)")

    # ─── Patient Summary ──────────────────────────────────────
    st.markdown("### 📋 Patient Summary")
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(f"**Age Group:** {age}")
        st.write(f"**Race:** {race}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Primary Diagnosis:** {diag_1}")
        st.write(f"**Insulin:** {insulin}")

    with summary_col2:
        st.write(f"**Time in Hospital:** {time_in_hospital} days")
        st.write(f"**Number of Diagnoses:** {number_diagnoses}")
        st.write(f"**Number of Medications:** {num_medications}")
        st.write(f"**Previous Inpatient Visits:** {number_inpatient}")
        st.write(f"**Service Utilization Score:** "
                 f"{number_outpatient + number_emergency + number_inpatient}")

    # ─── Disclaimer ───────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "⚠️ This tool is for research purposes only and should not "
        "replace clinical judgment. Always consult qualified medical "
        "professionals for patient care decisions."
    )

else:
    st.info("👈 Fill in patient details in the sidebar and click "
            "**Predict Readmission Risk** to get started.")

    # Show model info when no prediction yet
    st.markdown("### 📈 Model Information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "XGBoost")
    col2.metric("AUC-ROC", "0.653")
    col3.metric("Recall", "53%")
    col4.metric("Training Samples", "101,766")