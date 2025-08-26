import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from xgboost import XGBClassifier

# Import your preprocessing and feature engineering
from src.data_preprocessing import preprocess
from src.feature_engineering import add_features  # ensure your functions are modular


# ---------------- Utility functions ---------------- #

def metrics_report(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred)
    }

def get_scale_pos_weight(y):
    """Compute scale_pos_weight for XGBoost"""
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    return n_neg / max(n_pos, 1)

def best_threshold_under_fpr(y_true, y_proba, fpr_limit=0.005, cost_fp=1, cost_fn=25):
    """Return threshold minimizing cost under FPR constraint"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    best_cost = np.inf
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fpr_current = fp / max((fp + tn), 1)
        cost = fp*cost_fp + fn*cost_fn
        if fpr_current <= fpr_limit and cost < best_cost:
            best_cost = cost
            best_thresh = t
    return best_thresh

# ---------------- Training function ---------------- #

def train_model(X_train, y_train, X_val, y_val, feature_cols,
                models_dir="models", use_ensemble=True):

    os.makedirs(models_dir, exist_ok=True)
    trained_models = {}

    # 5a: Baselines
    print("Training Baselines...")

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train[feature_cols], y_train)
    trained_models["LR"] = lr

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train[feature_cols], y_train)
    trained_models["RF_baseline"] = rf

    # 5b: LightGBM
    print("Training LightGBM...")
    lgb_train = lgb.Dataset(X_train[feature_cols], label=y_train)
    lgb_val = lgb.Dataset(X_val[feature_cols], label=y_val)
    lgb_params = dict(
        objective="binary",
        metric=["auc", "average_precision"],
        is_unbalance=True,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        min_data_in_leaf=200,
        feature_fraction=0.6,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=3000
    )
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        verbose_eval=100,
        early_stopping_rounds=200
    )
    trained_models["LightGBM"] = lgb_model

    # 5b: XGBoost
    print("Training XGBoost...")
    xgb_pos_weight = get_scale_pos_weight(y_train)
    xgb_params = dict(
        objective="binary:logistic",
        eval_metric=["aucpr","auc"],
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        max_depth=6,
        max_bin=256,
        subsample=0.8,
        colsample_bytree=0.6,
        learning_rate=0.05,
        scale_pos_weight=xgb_pos_weight,
        n_estimators=3000,
        random_state=42
    )
    try:
        xgb_model = XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_train[feature_cols], y_train,
            eval_set=[(X_val[feature_cols], y_val)],
            verbose=False,
            early_stopping_rounds=200
        )
    except TypeError:
        print("⚠️ early_stopping_rounds not supported; fallback to CPU hist")
        xgb_params.update(dict(tree_method="hist", predictor="auto"))
        xgb_model = XGBClassifier(**xgb_params)
        xgb_model.fit(X_train[feature_cols], y_train)

    trained_models["XGBoost"] = xgb_model

    # 5c: Threshold sweep
    print("Computing best thresholds...")
    thresholds = {}
    for name, model in trained_models.items():
        if name == "LightGBM":
            val_proba = model.predict(X_val[feature_cols])
        else:
            val_proba = model.predict_proba(X_val[feature_cols])[:,1]
        best_thresh = best_threshold_under_fpr(y_val, val_proba)
        thresholds[name] = best_thresh
        print(f"{name} best threshold: {best_thresh:.4f}")

    # Optional ensemble
    if use_ensemble:
        print("Training Soft-Vote Ensemble (RF + XGB)...")
        ensemble = VotingClassifier(
            estimators=[("RF", rf), ("XGB", xgb_model)],
            voting="soft", n_jobs=-1
        )
        ensemble.fit(X_train[feature_cols], y_train)
        trained_models["Ensemble"] = ensemble
        thresholds["Ensemble"] = 0.5

    # Save models, feature list, thresholds
    print("Saving models and artifacts...")
    for name, model in trained_models.items():
        joblib.dump(model, os.path.join(models_dir, f"{name}.pkl"))

    pd.Series(feature_cols).to_csv(os.path.join(models_dir, "feature_list.csv"), index=False)
    with open(os.path.join(models_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f)

    print("All models saved in", models_dir)
    return trained_models, thresholds

# ---------------- Prediction function ---------------- #

def predict_model(df_new, model_name, models_dir="models"):
    # Preprocess & add features if not done
    df_new = preprocess(df_new)
    df_new = add_features(df_new)

    # Load model
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    model = joblib.load(model_path)

    # Load features
    feature_cols = pd.read_csv(os.path.join(models_dir, "feature_list.csv")).iloc[:,0].tolist()

    # Load thresholds
    with open(os.path.join(models_dir, "thresholds.json"), "r") as f:
        thresholds = json.load(f)
    threshold = thresholds.get(model_name, 0.5)

    # Predict probabilities
    if model_name == "LightGBM":
        proba = model.predict(df_new[feature_cols])
    else:
        proba = model.predict_proba(df_new[feature_cols])[:,1]

    preds = (proba >= threshold).astype(int)
    return proba, preds

# ---------------- CLI support ---------------- #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--predict", action="store_true", help="Predict from CSV")
    parser.add_argument("--input_csv", type=str, help="CSV path for prediction")
    parser.add_argument("--use_ensemble", action="store_true", help="Train soft-vote ensemble")
    args = parser.parse_args()

    if args.train:
        # Load processed features if available, else preprocess
        processed_path = "data/processed/transactions_features.csv"
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)
        else:
            raw_path = "data/raw/transactions.csv"  # adjust raw path
            df = pd.read_csv(raw_path)
            df = preprocess(df)
            df = add_features(df)
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv(processed_path, index=False)

        # Load feature list
        feature_list_path = os.path.join("models", "feature_list.csv")
        if os.path.exists(feature_list_path):
            feature_cols = pd.read_csv(feature_list_path).iloc[:,0].tolist()
        else:
            feature_cols = [c for c in df.columns if c not in ["TX_FRAUD","TRANSACTION_ID"]]

        X = df[feature_cols]
        y = df["TX_FRAUD"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
        train_model(X_train, y_train, X_val, y_val, feature_cols, use_ensemble=args.use_ensemble)

    if args.predict and args.input_csv:
        df_new = pd.read_csv(args.input_csv)
        proba, preds = predict_model(df_new, "XGBoost")
        print("Predictions:", preds)
