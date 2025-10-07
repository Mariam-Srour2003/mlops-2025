import argparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_parser():
    p = argparse.ArgumentParser(description="Feature Engineering")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p

def main():
    args = build_parser().parse_args()
    df = pd.read_csv(args.input)

    # Example: select numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    X = df[numeric_cols]
    y = df["target"] if "target" in df else None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_features = pd.DataFrame(X_scaled, columns=X.columns)
    if y is not None:
        df_features["target"] = y

    df_features.to_csv(args.output, index=False)
    print(f"Features saved to {args.output}")

if __name__ == "__main__":
    main()
