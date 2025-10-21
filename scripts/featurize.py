import argparse
import pandas as pd
from pathlib import Path

def build_parser():
    p = argparse.ArgumentParser(description="Feature engineering for Titanic dataset")
    p.add_argument("--input_train", required=True)
    p.add_argument("--input_test", required=True)
    p.add_argument("--output_train", required=True)
    p.add_argument("--output_test", required=True)
    return p


def engineer_features(df, is_train=True):
    # Create a FamilySize feature
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Base columns used for both train and test
    base_columns = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]

    # Add 'Survived' only if it's in the dataset (for training)
    if is_train and "Survived" in df.columns:
        columns = base_columns + ["Survived"]
    else:
        columns = base_columns

    # Keep only the selected columns
    df = df[columns]

    # Convert categorical variables into dummy/indicator variables
    df = pd.get_dummies(df, drop_first=True)

    return df


def main():
    args = build_parser().parse_args()

    # Ensure output directories exist
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    train = pd.read_csv(args.input_train)
    test = pd.read_csv(args.input_test)

    # Engineer features
    train = engineer_features(train, is_train=True)
    test = engineer_features(test, is_train=False)

    # Save feature-engineered datasets
    train.to_csv(args.output_train, index=False)
    test.to_csv(args.output_test, index=False)

    print(f"Features saved to {args.output_train} and {args.output_test}")


if __name__ == "__main__":
    main()
