from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestRegressor

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and preprocess dataset
file_path = r"C:\Users\Avi\Downloads\Vital Organs.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
for col in ["Gender", "Risk Category"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features
selected_features = [
    "Heart Rate", "Oxygen Saturation", "Systolic Blood Pressure", "Diastolic Blood Pressure", 
    "Derived_BMI", "Respiratory Rate", "Body Temperature", "Age", "Derived_HRV", 
    "Derived_Pulse_Pressure", "Derived_MAP"
]
X = df[selected_features]
y = df["Risk Category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train models
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb_model)], voting='soft')
ensemble_model.fit(X_train_scaled, y_train)

# Train SpO₂ and HRV Prediction Models
spo2_features = ["Respiratory Rate", "Heart Rate", "Body Temperature"]
hrv_features = ["Heart Rate", "Age", "Systolic Blood Pressure", "Diastolic Blood Pressure"]

spo2_model = RandomForestRegressor(n_estimators=100, random_state=42)
hrv_model = RandomForestRegressor(n_estimators=100, random_state=42)

spo2_model.fit(df.dropna(subset=spo2_features)[spo2_features], df.dropna(subset=spo2_features)["Oxygen Saturation"])
hrv_model.fit(df.dropna(subset=hrv_features)[hrv_features], df.dropna(subset=hrv_features)["Derived_HRV"])

# Function for predictions
def make_predictions(patient_data):
    prediction_messages = []
    prediction_values = {}
    
    if all(feature in patient_data for feature in spo2_features) and "Oxygen Saturation" not in patient_data:
        input_df = pd.DataFrame([patient_data])[spo2_features]
        predicted_spo2 = float(spo2_model.predict(input_df)[0])
        patient_data["Oxygen Saturation"] = predicted_spo2
        prediction_values["Oxygen Saturation"] = round(predicted_spo2, 3)
        prediction_messages.append(f"🔵 Predicted Oxygen Saturation: {predicted_spo2:.3f}%")
    
    if all(feature in patient_data for feature in hrv_features) and "Derived_HRV" not in patient_data:
        input_df = pd.DataFrame([patient_data])[hrv_features]
        predicted_hrv = float(hrv_model.predict(input_df)[0])
        patient_data["Derived_HRV"] = predicted_hrv
        prediction_values["Derived_HRV"] = round(predicted_hrv, 3)
        prediction_messages.append(f"🟢 Predicted Heart Rate Variability: {predicted_hrv:.3f}")
    
    return patient_data, prediction_messages, prediction_values

# Function for recommendations
def get_recommendations(patient_data):
    recommendations = []

    # Heart Rate Alerts
    if "Heart Rate" in patient_data:
        if patient_data["Heart Rate"] < 40:
            recommendations.append("🚨 **Critical:** Extremely low heart rate detected (<40 bpm). Seek emergency medical attention.")
        elif patient_data["Heart Rate"] < 60:
            recommendations.append("⚠️ Low heart rate detected. May indicate bradycardia. Consult a doctor.")
        elif patient_data["Heart Rate"] > 130:
            recommendations.append("🚨 **Critical:** Extremely high heart rate detected (>130 bpm). Seek immediate medical care.")
        elif patient_data["Heart Rate"] > 100:
            recommendations.append("⚠️ High heart rate detected. Stay hydrated and reduce stress.")

    # Oxygen Saturation Alerts
    if "Oxygen Saturation" in patient_data:
        if patient_data["Oxygen Saturation"] < 85:
            recommendations.append("🚨 **Critical:** Dangerously low oxygen levels detected (<85%). Immediate hospitalization needed!")
        elif patient_data["Oxygen Saturation"] < 95:
            recommendations.append("⚠️ Low oxygen levels detected. Ensure fresh air and rest.")

    # Respiratory Rate Alerts
    if "Respiratory Rate" in patient_data:
        if patient_data["Respiratory Rate"] < 8:
            recommendations.append("🚨 **Critical:** Severely low respiratory rate (<8 breaths/min). Seek urgent medical attention!")
        elif patient_data["Respiratory Rate"] < 12:
            recommendations.append("⚠️ Low respiratory rate. Monitor breathing patterns.")
        elif patient_data["Respiratory Rate"] > 30:
            recommendations.append("🚨 **Critical:** Dangerously high respiratory rate (>30 breaths/min). Seek emergency care!")
        elif patient_data["Respiratory Rate"] > 20:
            recommendations.append("⚠️ High respiratory rate. Check for stress or illness.")

    # Body Temperature Alerts
    if "Body Temperature" in patient_data:
        if patient_data["Body Temperature"] < 35:
            recommendations.append("🚨 **Critical:** Hypothermia detected (<35°C). Immediate warming and medical help required!")
        elif patient_data["Body Temperature"] < 36:
            recommendations.append("⚠️ Low body temperature detected. Stay warm.")
        elif patient_data["Body Temperature"] > 40:
            recommendations.append("🚨 **Critical:** Severe fever (>40°C). Seek emergency medical care immediately!")
        elif patient_data["Body Temperature"] > 38:
            recommendations.append("⚠️ High fever detected. Seek medical attention.")

    # Blood Pressure Alerts
    if "Systolic Blood Pressure" in patient_data and "Diastolic Blood Pressure" in patient_data:
        if patient_data["Systolic Blood Pressure"] > 180 or patient_data["Diastolic Blood Pressure"] > 120:
            recommendations.append("🚨 **Critical:** Hypertensive crisis detected (BP >180/120). Seek emergency care now!")
        elif patient_data["Systolic Blood Pressure"] < 80 or patient_data["Diastolic Blood Pressure"] < 50:
            recommendations.append("🚨 **Critical:** Dangerously low blood pressure (BP <80/50). Immediate medical help needed!")
        elif patient_data["Systolic Blood Pressure"] > 140 or patient_data["Diastolic Blood Pressure"] > 90:
            recommendations.append("⚠️ High blood pressure detected. Consider lifestyle changes.")
        elif patient_data["Systolic Blood Pressure"] < 90 or patient_data["Diastolic Blood Pressure"] < 60:
            recommendations.append("⚠️ Low blood pressure detected. Stay hydrated and avoid sudden movements.")

    # Age Consideration
    if "Age" in patient_data and patient_data["Age"] > 60:
        recommendations.append("🔶 Elderly patient detected. Regular check-ups recommended.")

    return recommendations


@app.route('/api/analyze', methods=['POST'])
def analyze_vitals():
    data = request.json
    patient_data = {}
    
    for feature in selected_features:
        if feature in data and data[feature] != '':
            try:
                patient_data[feature] = float(data[feature])
            except ValueError:
                return jsonify({'error': f'Invalid value for {feature}: {data[feature]}'}), 400
    
    updated_data, prediction_messages, prediction_values = make_predictions(patient_data)
    recommendations = get_recommendations(updated_data)
    
    return jsonify({
        'recommendations': recommendations,
        'predictions': prediction_messages,
        'predictionValues': prediction_values,
        'updatedData': {k: round(float(v), 4) for k, v in updated_data.items()}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
