# Fraud Transaction Detection using Machine Learning

An end-to-end machine learning system for detecting fraudulent financial transactions using advanced feature engineering, cost-sensitive modeling, and explainable AI.

---

## Problem Statement
Financial fraud causes massive losses in digital payment systems. This project builds a robust ML pipeline to identify fraudulent transactions while minimizing false positives.

---

## Models Used
- Logistic Regression
- Random Forest
- LightGBM
- XGBoost
- Soft Voting Ensemble

---

##  Project Structure

Fraud-Transaction-Detection/
│
├── app/ # Streamlit app
├── src/
│ ├── data_collection.py
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── model.py
│ └── evaluation.py
│
├── notebooks/ # EDA, modeling, evaluation
├── models/ # Saved models and artifacts
├── data/
│ ├── raw/
│ └── processed/
└── README.md



---

## How to Run
```bash
pip install -r requirements.txt
python src/model.py --train --use_ensemble
streamlit run app/app.py
