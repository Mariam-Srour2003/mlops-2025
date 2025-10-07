import argparse
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import json

def build_parser():
    p = argparse.ArgumentParser(description="Evaluate model")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=False)
    return p

def main():
    args = build_parser().parse_args()
    df = pd.read_csv(args.input)
    X = df.drop(columns="target")
    y = df["target"]

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"accuracy": acc}, f)
        print(f"Metrics saved to {args.output}")

if __name__ == "__main__":
    main()
