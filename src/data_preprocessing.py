import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Rename columns for consistency with your earlier plan
    df = df.rename(columns={
        'step': 'TX_STEP',
        'amount': 'TX_AMOUNT',
        'isFraud': 'TX_FRAUD',
        'nameOrig': 'CUSTOMER_ID',
        'nameDest': 'TERMINAL_ID'
    })
    
    # Convert step (hours) into days
    df['TX_DAY'] = df['TX_STEP'] // 24
    df['TX_HOUR'] = df['TX_STEP'] % 24
    
    # Keep only relevant columns (optional at this stage)
    # df = df[['TX_STEP', 'TX_AMOUNT', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_DAY', 'TX_HOUR', 'TX_FRAUD']]
    
    print("Processed dataframe shape:", df.shape)
    print("Fraud distribution:\n", df['TX_FRAUD'].value_counts(normalize=True))
    
    return df

if __name__ == "__main__":
    from data_collection import load_raw_data
    df = load_raw_data()
    df_processed = preprocess(df)
    print(df_processed.head())
