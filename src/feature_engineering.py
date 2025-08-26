import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    
    # CUSTOMER-LEVEL FEATURES
    
    customer_stats = (
        df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "cust_mean_amount",
            "std": "cust_std_amount",
            "count": "cust_tx_count"
        })
    )
    df = df.merge(customer_stats, on="CUSTOMER_ID", how="left")

    
    # TERMINAL-LEVEL FEATURES
   
    terminal_stats = (
        df.groupby("TERMINAL_ID")["TX_FRAUD"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={
            "mean": "terminal_fraud_ratio",
            "count": "terminal_tx_count"
        })
    )
    df = df.merge(terminal_stats, on="TERMINAL_ID", how="left")

    
    # TIME-BASED FEATURES
    
    # Rolling daily customer spend
    df["cust_daily_spend"] = (
        df.groupby(["CUSTOMER_ID", "TX_DAY"])["TX_AMOUNT"]
        .transform("sum")
    )
    
    return df


if __name__ == "__main__":
    from data_collection import load_raw_data
    from data_preprocessing import preprocess
    
    df = load_raw_data()
    df = preprocess(df)
    df = add_features(df)
    print(df.head())
