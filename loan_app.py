import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load models with error handling
try:
    model = joblib.load('loan_prediction.joblib')
    sc = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
    
    # Initialize label encoder with expected classes
    le.classes_ = np.array([0, 1])  # Explicitly set expected classes
    
    # Get the exact feature names and order from the trained model
    if hasattr(model, 'feature_names_in_'):
        expected_columns = list(model.feature_names_in_)
    else:
        # Fallback if feature names aren't available
        expected_columns = [
            'person_age', 'person_income', 'loan_amnt', 'loan_intent_DEBTCONSOLIDATION',
            'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
            'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_int_rate',
            'loan_percent_income', 'credit_score', 'previous_loan_defaults_on_file'
        ]
        
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# App title
st.title("💰 Loan Approval Prediction")
st.markdown("Predict whether a loan application will be approved based on applicant information")

# Input form
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        person_age = st.number_input("Age", 18, 100, 30)
        person_income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
        
    with col2:
        st.subheader("Loan Details")
        loan_amnt = st.number_input("Loan Amount ($)", 0, 1000000, 10000)
        loan_intent = st.selectbox("Loan Purpose", [
            "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", 
            "MEDICAL", "PERSONAL", "VENTURE"
        ])
    
    col3, col4 = st.columns(2)
    with col3:
        loan_int_rate = st.slider("Interest Rate (%)", 0.0, 20.0, 5.0)
        loan_percent_income = st.slider("Loan % of Income", 0.0, 100.0, 20.0)
    
    with col4:
        credit_score = st.slider("Credit Score", 300, 850, 700)
        prev_default = st.radio("Previous Defaults", ["No", "Yes"])
    
    submitted = st.form_submit_button("Predict Approval")

# Prediction logic
if submitted:
    try:
        # Prepare input data with correct feature names
        input_data = {
            'person_age': person_age,
            'person_income': person_income,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'credit_score': credit_score,
            'previous_loan_defaults_on_file': 1 if prev_default == "Yes" else 0
        }
        
        # One-hot encode loan intent
        for intent in ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", 
                      "MEDICAL", "PERSONAL", "VENTURE"]:
            input_data[f'loan_intent_{intent}'] = 1 if loan_intent == intent else 0
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data])
        
        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in input_df:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        
        # Scale numerical features
        num_cols = ['person_income', 'loan_amnt', 'credit_score']
        input_df[num_cols] = sc.transform(input_df[num_cols])
        
        # Encode categorical features
        input_df['previous_loan_defaults_on_file'] = le.transform(
            input_df['previous_loan_defaults_on_file']
        )
        
        # Ensure all columns are numeric
        input_df = input_df.astype(float)
        
        # Make prediction
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0]
        
        # Display results
        if prediction[0] == 1:
            st.success("### ✅ Loan Approved!")
            st.balloons()
        else:
            st.error("### ❌ Loan Denied")
        
        st.metric("Approval Probability", f"{proba[1]*100:.1f}%")
        
        # Show details
        with st.expander("Prediction Details"):
            st.write("**Input Features:**")
            st.dataframe(input_df)
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.error("Please ensure all input fields are filled correctly.")

# Add footer
st.markdown("---")
st.caption("Note: This is a demo application. Actual loan decisions may vary.")
