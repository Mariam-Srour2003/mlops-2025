from mlops_2025.preprocessing import Preprocessor
import pandas as pd

def main():
    # Load data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    # Initialize Preprocessor
    preprocessor = Preprocessor()
    df = preprocessor.process(train, test)
    
    # Continue with splitting and saving the data
    print("Preprocessing complete")
