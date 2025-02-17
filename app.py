from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('model.pkl')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load label encoders
try:
    encoders = joblib.load('label_encoders.pkl')
    print("✅ Encoders loaded successfully")
except Exception as e:
    print(f"❌ Error loading encoders: {e}")
    encoders = {}

# List of categorical features
categorical_features = [
    'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed',
    'MonthClaimed', 'Sex', 'MaritalStatus', 'Fault', 'PolicyType',
    'VehicleCategory', 'VehiclePrice', 'PoliceReportFiled', 'WitnessPresent',
    'AgentType', 'AddressChange-Claim', 'BasePolicy'
]

# List of numeric features
numeric_columns = [
    'WeekOfMonth', 'WeekOfMonthClaimed', 'Age', 'PolicyNumber', 'RepNumber',
    'Deductible', 'DriverRating', 'PastNumberOfClaims', 'NumberOfSuppliments',
    'NumberOfCars', 'Year'
]

# Define replacements for mapped values
replace_dict = {
    'more than 69,000': 69000,
    'more than 30': 30,
    'less than 1 year': 0.5,
    'none': 0,
    'None': 0,
    '': 0,  # Handle empty strings
    'new': 0
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/why')
def why():
    return render_template('why.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="⚠️ Error: Model not loaded correctly.")
    
    try:
        # Get input values from form
        input_data = {feature: request.form.get(feature, "").strip() for feature in request.form.keys()}
        print("📥 Received Input Data:", input_data)  # Debugging purpose

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert numeric columns to appropriate types
        for col in numeric_columns:
            if col in input_df.columns:
                input_df[col] = input_df[col].replace(replace_dict)
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # Encode categorical variables using stored encoders
        for col in categorical_features:
            if col in input_df.columns and col in encoders:
                encoder = encoders[col]
                if input_df[col].iloc[0] not in encoder.classes_:
                    input_df[col] = encoder.classes_[0]  # Assign default category if unseen
                input_df[col] = encoder.transform(input_df[col])

        # Ensure feature names match model training
        expected_features = model.feature_names_in_
        input_df = input_df.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)
        output = '🚨 Fraud Detected!' if prediction[0] == 1 else '✅ No Fraud'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return render_template('index.html', prediction_text=f'⚠️ Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
