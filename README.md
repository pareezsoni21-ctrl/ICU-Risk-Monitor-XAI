#ICU Risk Prediction: An Interpretable Clinical Decision Support System
##Project Overview:
This repository contains a clinical decision support tool designed to predict patient mortality risk in Intensive Care Units (ICU). The primary objective was to move beyond "black-box" predictions by implementing Explainable AI (XAI) techniques, ensuring that clinicians can verify the reasoning behind each risk score.
##Core Features
• Transparent Modeling: Developed using a Logistic Regression framework to ensure high interpretability and adherence to clinical safety standards.
• Explainable AI (XAI): Integrated SHAP (SHapley Additive exPlanations) to provide waterfall plots for every prediction. This allows users to see the exact contribution of variables like Mean Arterial Pressure (MAP) and Heart Rate to the overall risk.
• Interactive Interface: A Streamlit-based dashboard allowing for real-time data input and "what-if" scenario analysis for patient vitals.
• Clinical Metrics: Built-in logic to calculate physiological markers, including Pulse Pressure and Mean Arterial Pressure, from raw systolic and diastolic inputs.
##Technical Implementation
• Backend: Python, Scikit-Learn
• Interpretability: SHAP
• Frontend: Streamlit
• Environment: Managed via requirements.txt for reproducible deployment
##Data Source
The model was trained and validated using a public clinical dataset sourced from Kaggle, focused on ICU patient outcomes and vital sign telemetry.
