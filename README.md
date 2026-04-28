CU Risk Prediction: An Interpretable Clinical Decision Support System
This repository contains a clinical decision support tool designed to predict patient mortality risk in Intensive Care Units (ICU). The primary objective is to move beyond "black-box" predictions by implementing Explainable AI (XAI) techniques, ensuring that clinicians can verify the reasoning behind each risk score.

📂 Data Source
The model was trained and validated using the ICU Patient Outcome Prediction dataset, which provides electronic health record (EHR) data extracted from ICU environments to predict in-hospital mortality.  

Verified Dataset Link: Kaggle: ICU Patient Outcome Prediction

Dataset Content: Features include patient demographics (Age, Gender, Height), severity scores (SAPS-I, SOFA), and dynamic clinical variables (GCS, Glucose, Heart Rate, MAP, etc.).  

🚀 Core Features
Transparent Modeling: Built on a Logistic Regression framework to provide a baseline of high interpretability, ensuring the model meets clinical safety and audit standards.

Explainable AI (XAI): Integrated SHAP (SHapley Additive exPlanations) to generate waterfall plots for every individual prediction. This reveals exactly how much variables like Mean Arterial Pressure (MAP) or Heart Rate contributed to a specific risk score.

Interactive Dashboard: A Streamlit-based interface that allows clinicians to input patient data manually and perform "what-if" analyses to see how changing vitals impact the mortality risk.

Clinical Feature Engineering: Automated calculation of physiological markers, such as Pulse Pressure and Mean Arterial Pressure (MAP), directly from systolic and diastolic inputs.

🛠️ Technical Implementation
Backend: Python, Scikit-Learn

Interpretability: SHAP (Local and Global explanations)

Frontend: Streamlit

Environment: Reproducible via requirements.txt

📊 Methodology
The system processes raw clinical telemetry through three primary stages:

Feature Processing: Raw vitals are transformed into clinically significant markers (e.g., calculating MAP to assess tissue perfusion).

Risk Calibration: The model outputs a probability score (0.0 to 1.0) indicating the likelihood of in-hospital mortality.

Visualization: SHAP values decompose the prediction, highlighting "Red Flags" (factors increasing risk) and "Protective Factors" (factors decreasing risk) in real-time.

Disclaimer: This project is intended for research and educational purposes. It serves as a demonstration of how Explainable AI can be integrated into healthcare workflows and is not a substitute for professional medical judgment.
