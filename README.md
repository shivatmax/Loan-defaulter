# Loan Repayment Prediction Project

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Application Locally](#running-the-application-locally)
  - [API Endpoints](#api-endpoints)
  - [Using the API with `curl`](#using-the-api-with-curl)
- [Deployment](#deployment)
  - [Deploying on Hugging Face Spaces](#deploying-on-hugging-face-spaces)
- [Project Details](#project-details)
  - [Data Exploration and Feature Engineering](#data-exploration-and-feature-engineering)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The Loan Repayment Prediction Project aims to predict whether a borrower will repay or default on a loan based on user attributes and GPS data. This project involves data exploration, feature engineering, model training using machine learning algorithms, and deploying a predictive API.

**Key Objectives:**

- Analyze provided datasets and explore potential features.
- Train machine learning models to predict loan outcomes.
- Deploy the trained model through a simple API using Flask.
- Provide a user interface for new users to enter their details and get predictions.

---

## Data Sources

The project utilizes three main datasets:

1. **Loan Outcomes (`loan_outcomes`):**

   - `user_id`: Unique borrower identifier.
   - `application_at`: Timestamp of when the user applied for the loan.
   - `loan_outcome`: Whether the user repaid the loan in full (`repaid`) or defaulted (`defaulted`).

2. **GPS Fixes (`gps_fixes`):**

   - `user_id`: Unique borrower identifier.
   - `gps_fix_at`: Timestamp when GPS fix was collected.
   - `latitude`, `longitude`: Location coordinates.
   - Additional GPS-related attributes.

3. **User Attributes (`user_attributes`):**

   - `user_id`: Unique borrower identifier.
   - `age`: Age of the borrower at the time of the loan application.
   - `cash_incoming_30days`: Sum of money received by the borrower in the 30 days prior to the loan application.

**Note:** Data was accessed from a PostgreSQL database provided for the project.

---

## Project Structure

```
├── app.py                   # Flask application script
├── requirements.txt         # Dependencies required for the project
├── Procfile                 # Configuration for deployment
├── loan_default_model_rf.pkl  # Trained Random Forest model
├── scaler.pkl               # Scaler used during preprocessing
├── README.md                # Project documentation (this file)
```

---

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/loan-repayment-prediction.git
   cd loan-repayment-prediction
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Application Locally

1. **Ensure the trained model and scaler are present:**

   Make sure `loan_default_model_rf.pkl` and `scaler.pkl` are in the project directory.

2. **Run the Flask app:**

   ```bash
   python app.py
   ```

   The application will start on `http://0.0.0.0:7860/`.

3. **Test the API locally:**

   You can send a POST request to the `/predict` endpoint.

   ```bash
   curl -X POST "http://localhost:7860/predict" \
        -H "Content-Type: application/json" \
        -d @input.json
   ```

   Replace `@input.json` with your JSON file containing input data.

### API Endpoints

- `GET /` - Welcome message to confirm the API is running.
- `POST /predict` - Predicts loan outcome based on user input.

#### Request Body for `/predict`

The `/predict` endpoint expects a JSON payload with the following fields:

- `age` (int): Age of the borrower.
- `cash_incoming_30days` (float): Cash incoming in the last 30 days.
- `gps_fix_count` (int, optional): Number of app opens (GPS fixes). Defaults to 0 if not provided.
- `unique_locations_count` (int, optional): Number of unique locations visited. Defaults to 0 if not provided.
- `avg_time_between_opens` (float, optional): Average time between app opens in seconds. Defaults to 0.0 if not provided.
- `night_usage_ratio` (float, optional): Ratio of nighttime app usage (0 to 1). Defaults to 0.0 if not provided.
- `num_clusters` (int, optional): Number of significant locations (clusters). Defaults to 0 if not provided.

#### Example Request

```json
{
  "age": 30,
  "cash_incoming_30days": 5000.0,
  "gps_fix_count": 15,
  "unique_locations_count": 5,
  "avg_time_between_opens": 3600.0,
  "night_usage_ratio": 0.2,
  "num_clusters": 2
}
```

#### Example Response

```json
{
  "prediction": "Repaid",
  "probability": 0.85
}
```

### Using the API with `curl`

Save the input data in a file named `input.json` and use the following command:

```bash
curl -X POST "http://localhost:7860/predict" \
     -H "Content-Type: application/json" \
     -d @input.json
```

---

## Deployment

### Deploying on Hugging Face Spaces

To deploy your Flask API on Hugging Face Spaces:

1. **Create a new Space:**

   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
   - Name your Space (e.g., `loan-default-api`).
   - Set the **SDK** to **Streamlit** (we'll adjust this with a `Procfile`).

2. **Add a `Procfile`:**

   Create a file named `Procfile` with the following content:

   ```
   web: waitress-serve --port=$PORT app:app
   ```

3. **Push your code to the Space:**

   - **Option 1:** Upload files via the web interface.
   - **Option 2:** Use Git to clone the repository, copy your files, and push back to Hugging Face.

4. **Monitor deployment:**

   Check the **Logs** tab on your Space to ensure the app builds and runs successfully.

5. **Access your API:**

   Your API will be available at:

   ```
   https://your-username-loan-default-api.hf.space
   ```

6. **Test the API:**

   Use `curl` to send requests to your deployed API, similar to how you would test it locally.

---

## Project Details

### Data Exploration and Feature Engineering

- **Data Exploration:**
  - Checked for missing values and ensured data integrity.
  - Analyzed the distribution of loan outcomes (balanced dataset with 200 defaulted and 200 repaid loans).
  - Performed statistical analysis on user attributes such as age and cash incoming.

- **Feature Engineering:**
  - **User Attributes:**
    - Applied log transformation to `cash_incoming_30days` to handle skewness.
    - Created income brackets (`Low`, `Medium`, `High`, `Very High`) based on cash incoming.

  - **GPS-Based Features:**
    - Calculated `gps_fix_count` as the number of app opens.
    - Determined `unique_locations_count` by counting unique latitude and longitude pairs.
    - Computed `avg_time_between_opens` to understand user engagement.
    - Calculated `night_usage_ratio` as the proportion of app usage occurring at night.
    - Performed cluster analysis to identify significant locations (`num_clusters`).

### Model Training and Evaluation

- **Models Used:**
  - **Random Forest Classifier**
  - **CatBoost Classifier**
  - **Neural Network (Multi-layer Perceptron)**

- **Hyperparameter Tuning:**
  - Used `GridSearchCV` for hyperparameter tuning and cross-validation.
  - Optimized key parameters for each model to improve performance.

- **Model Evaluation:**
  - Evaluated models using metrics such as accuracy, precision, recall, F1-score, and ROC AUC score.
  - Plotted ROC curves to compare model performance.
  - Analyzed feature importance to interpret model predictions.

- **Model Selection:**
  - Selected the **Random Forest Classifier** as the deployment model based on performance and interpretability.

---

## Dependencies

The project requires the following packages:

- **Flask**: Web framework used for the API.
- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical computing.
- **scikit-learn**: Machine learning library.
- **joblib**: Model persistence.
- **waitress**: WSGI server for serving the Flask app.

Refer to `requirements.txt` for exact versions.

