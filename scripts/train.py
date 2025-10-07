import argparse
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

def build_parser():
    p = argparse.ArgumentParser(description="Train model")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p

def main():
    args = build_parser().parse_args()
    df = pd.read_csv(args.input)
    X = df.drop(columns="target")
    y = df["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()
