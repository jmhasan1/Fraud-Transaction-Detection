import pandas as pd

def load_raw_data(path="data/raw/fraudulent_transactions.csv"):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
