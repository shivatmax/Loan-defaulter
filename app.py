from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

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

# Load the saved model and scaler
model = joblib.load('loan_default_model_rf.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize FastAPI app
app = FastAPI()

# Define the request body using Pydantic
class UserInput(BaseModel):
    age: int
    cash_incoming_30days: float
    gps_fix_count: int = 0
    unique_locations_count: int = 0
    avg_time_between_opens: float = 0.0
    night_usage_ratio: float = 0.0
    num_clusters: int = 0

@app.post("/predict")
def predict_loan_outcome(input_data: UserInput):
    # Convert input data to a dictionary
    data = input_data.dict()
    
    # Create a DataFrame
    input_df = pd.DataFrame([data])
    
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
    numerical_cols = [
        'age',
        'log_cash_incoming_30days',
        'gps_fix_count',
        'unique_locations_count',
        'avg_time_between_opens',
        'night_usage_ratio',
        'num_clusters'
    ]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Ensure the input_df has all the required columns
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    outcome = 'Repaid' if prediction[0] == 1 else 'Defaulted'
    proba = prediction_proba[0][prediction[0]]
    
    return {
        'prediction': outcome,
        'probability': float(proba)
    }