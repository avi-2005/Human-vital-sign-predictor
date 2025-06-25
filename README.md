# Human-vital-sign-predictor
# 🩺 Human Vital Sign Predictor

A full-fledged **desktop application** with a user-friendly **Tkinter GUI** and a powerful **Flask backend** that predicts **key human vital signs** and assesses potential health risks using machine learning models. Designed to support quick, intelligent screening and deliver actionable health recommendations.

---

## 🚀 Key Features

- ✅ Predicts vital signs like **Oxygen Saturation** and **Heart Rate Variability (HRV)** using regression models  
- 🧠 Classifies **risk category** of a patient using ensemble ML models  
- 🪟 Clean, animated **Tkinter interface** for local use (no browser needed)  
- 📡 Connects to a local **Flask API** for real-time predictions and health analysis  
- 🔔 Generates **critical alerts and medical recommendations** based on inputs  
- 🎞 Includes animated health-themed GIF for a friendly UI  

---

## 🧱 Tech Stack

| Layer       | Technologies |
|-------------|--------------|
| Frontend    | Python, Tkinter, PIL (for GIF animation) |
| Backend     | Flask, Flask-CORS |
| ML Models   | Scikit-Learn (RandomForest, GradientBoosting, VotingClassifier) |
| Data        | CSV from clinical dataset (`Vital Organs.csv`) |
| Others      | Requests, Joblib, Pandas, NumPy |

---

## 💻 Application Architecture

GUI (Tkinter)
│

├── Inputs from user (vitals)

│
▼

Backend API (Flask)

├── Predict missing vitals (SpO₂, HRV) via regression

├── Classify Risk Category using ensemble ML

├── Generate health alerts & recommendations
▼

GUI (Displays predictions + suggestions)

---

## 🩺 Inputs Collected

- Heart Rate  
- Oxygen Saturation  
- Systolic Blood Pressure  
- Diastolic Blood Pressure  
- Body Temperature  
- Respiratory Rate  
- Derived BMI  
- Derived HRV  
- Derived Pulse Pressure  
- Derived MAP  
- Age  

⚠️ **Partial inputs are accepted.** Missing values for **SpO₂** or **HRV** will be predicted automatically.

---

## 📊 Outputs Generated

- 🔵 Predicted Oxygen Saturation (if not provided)  
- 🟢 Predicted Heart Rate Variability (HRV)  
- 🟡 Classified **Risk Category** (Low / Moderate / High)  
- 🔴 Health recommendations (e.g. “Hypertensive crisis”, “Low oxygen level”, “Fever”)  
- 📋 Clean display of updated input values and calculated predictions  

---

## 🧪 Model Info

- **Ensemble Classifier:** VotingClassifier with:  
  - RandomForestClassifier (n=200, max_depth=10)  
  - GradientBoostingClassifier (n=200, LR=0.05)  
- **Regressors:**  
  - RandomForestRegressor for SpO₂ and HRV  
- **Preprocessing:**  
  - Label Encoding (for Gender, Risk Category)  
  - Standard Scaling of features  

---

## 📁 Folder Structure

human-vital-sign-predictor/
├── app1.py # Flask backend (API and ML models)

├── gui1.py # Tkinter frontend with animated GIF and form

├── Vital Organs.csv # Clinical dataset

├── gif heart.gif # Health-themed animation

├── requirements.txt # Python dependencies

└── README.md # This file

---

## ⚙️ How to Run

1. Clone the repository
git clone https://github.com/your-username/human-vital-sign-predictor.git
cd human-vital-sign-predictor
2. Install required packages
bash
Copy
Edit
pip install -r requirements.txt
3. Start the Flask backend
bash
Copy
Edit
python app1.py
4. Start the GUI application
bash
Copy
Edit
python gui1.py
✅ Make sure the backend is running before launching the GUI.

📦 requirements.txt

txt

Copy

Edit

flask

flask-cors

requests

pandas

numpy

scikit-learn

joblib

pillow

🛠️ Future Improvements

Add PDF export for patient reports

Integrate additional vitals like ECG signals or stress index

Deploy API on a remote server and convert GUI to a web app

Include real-time sensor input from IoT devices

🙌 Credits
Dataset: Internal or public health dataset (e.g. [Kaggle/UCI])

Developed by: Avi Agrawal

Email: aviagrawal2005@gmail.com

LinkedIn: https://www.linkedin.com/in/avi-agrawal-b47077263/

⭐️ Show Your Support
If this project helped or inspired you, please star this repo and consider sharing it with others in the healthtech or data science community.
