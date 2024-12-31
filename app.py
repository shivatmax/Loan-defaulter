from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('loan_default_model_rf.pkl')
scaler = joblib.load('scaler.pkl')

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

numerical_cols = [
    'age',
    'log_cash_incoming_30days',
    'gps_fix_count',
    'unique_locations_count',
    'avg_time_between_opens',
    'night_usage_ratio',
    'num_clusters'
]

# Define the home route
@app.route('/')
def index():
    return "Welcome to the Loan Repayment Prediction API!"

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict_loan_outcome():
    # Get the JSON data from the request
    input_data = request.get_json(force=True)
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocessing
    # Log transformation
    input_df['log_cash_incoming_30days'] = np.log1p(input_df['cash_incoming_30days'])
    
    # Income bracket
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

    # Handle missing brackets
    for bracket in ['Medium', 'High', 'Very High']:
        col_name = f'income_bracket_{bracket}'
        if col_name not in input_df.columns:
            input_df[col_name] = 0
    
    # Fill missing GPS features with zeros if necessary
    gps_feature_cols = ['gps_fix_count', 'unique_locations_count', 'avg_time_between_opens', 'night_usage_ratio', 'num_clusters']
    input_df[gps_feature_cols] = input_df[gps_feature_cols].fillna(0)
    
    # Scale numerical features
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Ensure the input_df has all the required columns
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    outcome = 'Repaid' if prediction[0] == 1 else 'Defaulted'
    proba = prediction_proba[0][prediction[0]]
    
    # Return the prediction as JSON
    return jsonify({'prediction': outcome, 'probability': float(proba)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
