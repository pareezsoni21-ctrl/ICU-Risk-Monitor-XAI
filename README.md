# ICU Risk Prediction: Interpretable Clinical Decision Support System

## Overview
This project focuses on building a clinical decision support system to predict in-hospital mortality risk for patients admitted to Intensive Care Units (ICUs). The system is designed with a strong emphasis on interpretability, ensuring that predictions are not only accurate but also explainable.
In clinical environments, understanding why a model makes a prediction is as important as the prediction itself. This project addresses that need by combining a transparent modeling approach with techniques that allow detailed analysis of feature contributions. The overall goal is to support informed decision-making rather than relying on opaque, black-box systems.

## Dataset
The model is trained on ICU patient data consisting of 3,600 records and 120 initial features. These include:
Demographic variables such as age and gender
Clinical severity scores including SAPS-I and SOFA
Physiological and laboratory measurements such as heart rate, pH, glucose, and blood pressure-related features
ICU admission indicators (e.g., CCU, CSRU, SICU)

The target variable represents in-hospital mortality, with a noticeable class imbalance between survivors and non-survivors.

## Data Preprocessing
The dataset required extensive preprocessing to ensure reliability and consistency.
Columns with more than 70% missing values were removed to reduce noise and improve model stability
Remaining missing values were imputed using the median, preserving the distribution of clinical variables
The dataset was split into training and testing sets using an 80–20 ratio
After preprocessing, the feature space was reduced from 120 to 68 meaningful variables.

## Modeling Approach
The primary model used in this project is Logistic Regression. This choice was made to maintain interpretability, as each feature directly contributes to the final prediction through its coefficient.

Two configurations were evaluated:
Standard Logistic Regression
Logistic Regression with class balancing

The imbalanced nature of the dataset required special attention. By applying class weighting, the model became more sensitive to minority class cases (patients at risk of mortality), which is critical in medical applications.

## Model Performance
The initial model achieved high overall accuracy but showed poor recall for mortality cases. This indicated that the model was biased toward predicting survival.
After applying class weighting:
Recall for high-risk patients improved significantly
The model became more effective at identifying critical cases
A trade-off was observed in overall accuracy, which is acceptable in healthcare scenarios where missing high-risk patients is more critical than false alarms
Threshold tuning was also performed to further adjust the balance between precision and recall, allowing flexibility depending on the clinical objective.

## Alternative Model Comparison
An XGBoost model was also implemented to compare performance.
XGBoost provided competitive accuracy
However, it did not significantly outperform Logistic Regression in identifying high-risk patients
Given the increased complexity and reduced interpretability, Logistic Regression was retained as the primary model
This decision reinforces the project's focus on transparency and clinical usability.

## Feature Importance and Interpretability
Feature importance was analyzed using the coefficients of the Logistic Regression model.
Some of the most influential features included:
pH levels (first and last measurements)
Glasgow Coma Scale (GCS)
Magnesium levels
Creatinine levels
ICU unit indicators such as CCU and CSRU
These features align with clinical expectations, supporting the validity of the model.

The project also integrates SHAP (SHapley Additive Explanations) to provide detailed, patient-level explanations. This allows each prediction to be broken down into contributing factors, making the model’s decisions transparent and interpretable.

## Deployment
The trained model is saved using joblib and integrated into a Streamlit application.

The Streamlit interface allows users to:
Input patient data manually
Generate real-time mortality risk predictions
Analyze how different clinical variables influence the outcome

The application provides an interactive environment for exploring model behavior and understanding patient risk profiles.

## Use Case
This system demonstrates how interpretable machine learning can be applied in healthcare to assist in risk assessment. It is particularly useful in scenarios where understanding the reasoning behind predictions is essential.

The project can serve as a foundation for:
Clinical decision support tools
Healthcare analytics systems
Research in explainable AI for medicine

## Disclaimer
This project is intended for research and educational purposes only. It is not designed for clinical deployment and should not be used as a substitute for professional medical judgment.
