import pandas as pd
import os

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Rename columns for consistency
    df = df.rename(columns={
        'step': 'TX_STEP',
        'amount': 'TX_AMOUNT',
        'isFraud': 'TX_FRAUD',
        'nameOrig': 'CUSTOMER_ID',
        'nameDest': 'TERMINAL_ID'
    })
    
    # Convert step (hours) into days and hours
    df['TX_DAY'] = df['TX_STEP'] // 24
    df['TX_HOUR'] = df['TX_STEP'] % 24
    
    print("Processed dataframe shape:", df.shape)
    print("Fraud distribution:\n", df['TX_FRAUD'].value_counts(normalize=True))
    
    return df

if __name__ == "__main__":
    from data_collection import load_raw_data
    
    # Load raw data
    df = load_raw_data()
    
    # Preprocess
    df_processed = preprocess(df)
    print(df_processed.head())
    
    # --------------------------
    # Save processed features
    # --------------------------
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "transactions_features.csv")
    df_processed.to_csv(processed_path, index=False)
    print(f"Processed features saved to {processed_path}")
    
    # --------------------------
    # Save feature list
    # --------------------------
    feature_cols = [col for col in df_processed.columns if col != "TX_FRAUD"]
    os.makedirs("models", exist_ok=True)
    feature_list_path = os.path.join("models", "feature_list.csv")
    pd.DataFrame(feature_cols).to_csv(feature_list_path, index=False)
    print(f"Feature list saved to {feature_list_path}")
