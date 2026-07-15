from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained components
with open('model.pkl', 'rb') as m_file:
    model = pickle.load(m_file)
with open('scaler.pkl', 'rb') as s_file:
    scaler = pickle.load(s_file)
with open('encoders.pkl', 'rb') as e_file:
    encoders = pickle.load(e_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect values from application form matching dataset expectations
        # Core standard inputs: Gender, Car, Realty, Children, Income, Income Type, Education, Family Status, Housing, Age, Employed Days
        
        raw_data = {
            'CODE_GENDER': request.form['gender'],
            'FLAG_OWN_CAR': request.form['car'],
            'FLAG_OWN_REALTY': request.form['realty'],
            'CNT_CHILDREN': int(request.form['children']),
            'AMT_INCOME_TOTAL': float(request.form['income']),
            'NAME_INCOME_TYPE': request.form['income_type'],
            'NAME_EDUCATION_TYPE': request.form['education'],
            'NAME_FAMILY_STATUS': request.form['family_status'],
            'NAME_HOUSING_TYPE': request.form['housing_type'],
            'DAYS_BIRTH': -int(request.form['age']) * 365,       # Dataset keeps days relative to today (negative)
            'DAYS_EMPLOYED': -int(request.form['experience']) * 365,
            'FLAG_MOBIL': 1,
            'FLAG_WORK_PHONE': int(request.form['work_phone']),
            'FLAG_PHONE': int(request.form['phone']),
            'FLAG_EMAIL': int(request.form['email']),
            'OCCUPATION_TYPE': request.form['occupation'],
            'CNT_FAM_MEMBERS': int(request.form['family_size'])
        }
        
        input_df = pd.DataFrame([raw_data])
        
        # Transform categorical elements using saved configurations
        for col in encoders.keys():
            if col in input_df.columns:
                try:
                    input_df[col] = encoders[col].transform(input_df[col])
                except ValueError:
                    # Fallback default value for unseen testing elements
                    input_df[col] = 0

        # Run normalization transformations
        scaled_input = scaler.transform(input_df)
        
        # Run calculation
        prediction = model.predict(scaled_input)[0]
        
        # Structure final status assessment text output
        if prediction == 0:
            result_text = "🎉 Congratulations! Your Credit Card Application is APPROVED."
            result_class = "approved"
        else:
            result_text = "❌ Sorry, Your Credit Card Application has been REJECTED based on risk analysis."
            result_class = "rejected"
            
        return render_template('index.html', prediction_text=result_text, result_style=result_class)

if __name__ == "__main__":
    app.run(debug=True)