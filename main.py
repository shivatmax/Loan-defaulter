import streamlit as st
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load('scaler.pkl')
model = joblib.load('loan_default_model_cb.pkl') 


# Define the feature columns
feature_cols = [
    'age',
    'log_cash_incoming_30days', 
    'gps_fix_count',
    'unique_locations_count',
    'avg_time_between_opens',
    'night_usage_ratio',
    'num_clusters',
    'income_bracket_Medium',
    'income_bracket_High',
    'income_bracket_Very High'
]

# Define numerical and categorical columns
numerical_cols = [
    'age',
    'log_cash_incoming_30days',
    'gps_fix_count', 
    'unique_locations_count',
    'avg_time_between_opens',
    'night_usage_ratio',
    'num_clusters'
]

categorical_cols = [
    'income_bracket_Medium',
    'income_bracket_High',
    'income_bracket_Very High'
]

# Function to preprocess user input
def preprocess_input(user_input):
    # Create DataFrame
    input_df = pd.DataFrame([user_input])

    # Handle missing GPS features if any
    gps_feature_cols = [
        'gps_fix_count',
        'unique_locations_count', 
        'avg_time_between_opens',
        'night_usage_ratio',
        'num_clusters'
    ]
    input_df[gps_feature_cols] = input_df[gps_feature_cols].fillna(0)

    # Log transformation for cash_incoming_30days
    input_df['log_cash_incoming_30days'] = np.log1p(input_df['cash_incoming_30days'])

    # Income brackets
    cash_incoming = input_df['cash_incoming_30days'].values[0]
    if cash_incoming < 2000:
        income_bracket = 'Low'
    elif cash_incoming < 5000:
        income_bracket = 'Medium'
    elif cash_incoming < 10000:
        income_bracket = 'High'
    else:
        income_bracket = 'Very High'

    # One-hot encoding for income bracket
    for bracket in ['Medium', 'High', 'Very High']:
        col_name = f'income_bracket_{bracket}'
        input_df[col_name] = 1 if income_bracket == bracket else 0

    # Scale numerical features
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Select the columns in the correct order
    input_df = input_df[feature_cols]

    return input_df

# Streamlit app
def main():
    st.title("Loan Repayment Prediction")

    st.write("""
    Enter your details to check the loan repayment prediction.
    """)

    # User inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    cash_incoming_30days = st.number_input("Cash Incoming in Last 30 Days (KES)", min_value=0.0, value=5000.0)

    st.write("### GPS-based Features (optional)")

    gps_fix_count = st.number_input("Number of App Opens (GPS Fix Count)", min_value=0, value=10)
    unique_locations_count = st.number_input("Unique Locations Visited", min_value=0, value=5)
    avg_time_between_opens = st.number_input("Average Time Between App Opens (seconds)", min_value=0.0, value=3600.0)
    night_usage_ratio = st.slider("Nighttime Activity Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=0.2)
    num_clusters = st.number_input("Number of Significant Locations (Clusters)", min_value=0, value=2)

    # Prepare user input
    user_input = {
        'age': age,
        'cash_incoming_30days': cash_incoming_30days,
        'gps_fix_count': gps_fix_count,
        'unique_locations_count': unique_locations_count,
        'avg_time_between_opens': avg_time_between_opens,
        'night_usage_ratio': night_usage_ratio,
        'num_clusters': num_clusters
    }

    if st.button("Predict Loan Outcome"):
        try:
            # Preprocess input
            input_data = preprocess_input(user_input)

            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            # Display result
            outcome = 'Repaid' if prediction[0] == 1 else 'Defaulted'
            proba = prediction_proba[0][prediction[0]]

            st.write(f"### Prediction: {outcome}")
            st.write(f"Probability: {proba:.2f}")

            # Optionally, display the input data
            st.write("#### Input Data:")
            st.write(input_data)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == '__main__':
    main()