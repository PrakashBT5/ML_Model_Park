from flask import Flask, render_template, request
import pickle
import pandas as pd
from datetime import datetime

# Load trained model and label encoder
with open('rf_parking_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load original data to extract unique locations
df = pd.read_csv('D:/Park_Pred_Model/brim_data.csv')
locations = sorted(df['SystemCodeNumber'].unique())

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', locations=locations, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']

    now = datetime.now()
    hour = now.hour
    day_of_week = now.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0

    encoded_location = le.transform([location])[0]
    features = [[encoded_location, hour, day_of_week, is_weekend]]
    probs = model.predict_proba(features)[0]
    prediction = 1 if probs[1] >= 0.9 else 0
    result = "Available ✅" if prediction == 1 else "Not Available ❌"

    return render_template(
        'index.html',
        locations=locations,
        prediction=result,
        selected_location=location  # 🔥 Pass this too
    )



if __name__ == '__main__':
    app.run(debug=True)
