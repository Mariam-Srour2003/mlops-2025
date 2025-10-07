import argparse
import pandas as pd

def build_parser():
    p = argparse.ArgumentParser(description="Preprocess CSV data")
    p.add_argument("--input", required=True, help="Raw CSV path")
    p.add_argument("--output", required=True, help="Processed CSV path")
    return p

def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)

    df = df.drop(columns=["unnecessary_column"], errors="ignore")
    df = df.fillna(0)

    df.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}")

if __name__ == "__main__":
    main()
