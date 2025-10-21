Write-Host "=== Step 1: Preprocessing ==="
uv run python scripts/preprocess.py --train_path data/raw/train.csv --test_path data/raw/test.csv --output_train data/processed/train_processed.csv --output_test data/processed/test_processed.csv

Write-Host "=== Step 2: Feature Engineering ==="
uv run python scripts/featurize.py --input_train data/processed/train_processed.csv --input_test data/processed/test_processed.csv --output_train data/features/train_features.csv --output_test data/features/test_features.csv

Write-Host "=== Step 3: Training Model ==="
uv run python scripts/train.py --input data/features/train_features.csv --output_model models/log_reg.pkl

Write-Host "=== Step 4: Evaluating Model ==="
uv run python scripts/evaluate.py --model models/log_reg.pkl --data data/features/train_features.csv --output_metrics metrics/train_eval.json

Write-Host "=== Step 5: Predicting ==="
uv run python scripts/predict.py --model models/log_reg.pkl --input data/features/test_features.csv --output predictions/test_predictions.csv

Write-Host "✅ Pipeline completed successfully!"
