import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import shap
import argparse

 
# Metrics and utility functions
 
def metrics_report(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred)
    }

 
# Load model and artifacts
 
def load_model_artifacts(model_name, models_dir="models"):
    # Load model
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    model = joblib.load(model_path)
    
    # Load features
    feature_cols = pd.read_csv(os.path.join(models_dir, "feature_list.csv")).iloc[:,0].tolist()
    
    # Load thresholds
    thresholds = np.load(os.path.join(models_dir, "thresholds.npy"), allow_pickle=True).item()
    threshold = thresholds.get(model_name, 0.5)
    
    return model, feature_cols, threshold

 
# Plotting functions
 
def plot_confusion_matrix(cm, model_name, threshold):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix (Threshold={threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_roc_curve(y_true, y_proba, model_name):
    roc_auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.show()

def plot_pr_curve(y_true, y_proba, model_name):
    pr_auc = average_precision_score(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.legend()
    plt.show()

def plot_cost_vs_threshold(y_true, y_proba, threshold, model_name, fpr_limit=0.005, cost_fp=1, cost_fn=25):
    costs = []
    threshold_list = np.linspace(0,1,1000)
    for t in threshold_list:
        y_p = (y_proba >= t).astype(int)
        tn = np.sum((y_true==0)&(y_p==0))
        fp = np.sum((y_true==0)&(y_p==1))
        fn = np.sum((y_true==1)&(y_p==0))
        cost = fp*cost_fp + fn*cost_fn
        costs.append(cost)

    plt.figure(figsize=(6,5))
    plt.plot(threshold_list, costs)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f"Selected Threshold={threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Cost")
    plt.title(f"{model_name} Cost vs Threshold")
    plt.legend()
    plt.show()

 
# SHAP Explainability
 
def shap_explain(model, X, model_name, df, y_pred, top_n=5):
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Global importance
    shap.summary_plot(shap_values, X, plot_type="bar")
    shap.summary_plot(shap_values, X)
    
    # Local explanation for top flagged frauds
    fraud_idx = np.where(y_pred==1)[0]
    sample_idx = fraud_idx[:top_n]
    for idx in sample_idx:
        print(f"\nTransaction ID: {df.iloc[idx]['TRANSACTION_ID']}")
        shap.force_plot(explainer.expected_value, shap_values[idx], X.iloc[idx], matplotlib=True)

 
# Main evaluation function
 
def evaluate_model(df, model_name, models_dir="models"):
    model, feature_cols, threshold = load_model_artifacts(model_name, models_dir)
    X = df[feature_cols]
    y = df["TX_FRAUD"]
    
    # Predict probabilities
    if model_name == "LightGBM":
        y_proba = model.predict(X)
    else:
        y_proba = model.predict_proba(X)[:,1]
    
    y_pred = (y_proba >= threshold).astype(int)
    
    # Metrics
    metrics = metrics_report(y, y_proba, threshold)
    print(f"{model_name} Metrics:")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")
    print("Classification Report:\n", metrics['report'])
    print("Confusion Matrix:\n", metrics['confusion_matrix'])
    
    # Plots
    plot_confusion_matrix(metrics['confusion_matrix'], model_name, threshold)
    plot_roc_curve(y, y_proba, model_name)
    plot_pr_curve(y, y_proba, model_name)
    plot_cost_vs_threshold(y, y_proba, threshold, model_name)
    
    # SHAP explainability
    shap_explain(model, X, model_name, df, y_pred)

 
# CLI
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., XGBoost, LightGBM, Ensemble)")
    parser.add_argument("--input_csv", type=str, default="data/processed/transactions_features.csv", help="CSV path with features")
    args = parser.parse_args()
    
    df_eval = pd.read_csv(args.input_csv)
    evaluate_model(df_eval, args.model)
