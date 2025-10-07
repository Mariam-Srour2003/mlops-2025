import argparse
import pandas as pd
import pickle

def build_parser():
    p = argparse.ArgumentParser(description="Predict with trained model")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p

def main():
    args = build_parser().parse_args()
    df = pd.read_csv(args.input)

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(df)
    pd.DataFrame({"prediction": predictions}).to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
