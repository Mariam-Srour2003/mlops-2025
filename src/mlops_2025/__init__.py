import pandas as pd
from mlops_2025.preprocessing import Preprocessor
from mlops_2025.features import FeatureComputer
from mlops_2025.models import LogisticRegressionModel, RandomForestModel, XGBoostModel

def load_data():
    # Load your dataset here
    # For example, use pandas to load data
    data = pd.read_csv("data/titanic.csv")
    return data

def main():
    # Load data
    raw_data = load_data()
    
    # Step 1: Preprocess the data
    preprocessor = Preprocessor()
    processed_data = preprocessor.process(raw_data)
    
    # Step 2: Feature Engineering
    feature_engineer = FeatureComputer()
    features = feature_engineer.create_features(processed_data)
    
    # Step 3: Initialize the model
    model = LogisticRegressionModel()  # You can swap with RandomForestModel or XGBModel
    # Or, if you want to run both, you can:
    # model = RandomForestModel()  # Or XGBModel()

    # Step 4: Train the model
    model.train(features)
    
    # Step 5: Evaluate the model
    evaluation_results = model.evaluate()
    print("Model Evaluation Results:", evaluation_results)
    
    # Step 6: Make Predictions
    predictions = model.predict(features)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
