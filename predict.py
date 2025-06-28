import pandas as pd
import joblib
from src.preprocessing import preprocess_data
from src.feature_engineering import add_features

def generate_submission(model, test_df, sample_submission_path):
    predictions = model.predict(test_df)
    submission = pd.read_csv(sample_submission_path)
    submission['Rating'] = predictions
    submission.to_csv("submission.csv", index=False)
    print("✅ submission.csv created!")

# Load preprocessed test data
test_df = pd.read_csv("data/test_processed.csv")

# Load the trained model
model = joblib.load("models/final_model.pkl")

# Ensure test_df columns match training features
# (only needed if you're using scikit-learn >=1.0, which provides `feature_names_in_`)
if hasattr(model, 'feature_names_in_'):
    test_df = test_df[model.feature_names_in_]
else:
    # OR: use saved feature column list from training (uncomment if you saved it)
    # feature_cols = pd.read_csv("models/feature_columns.csv", header=None)[0].tolist()
    # test_df = test_df[feature_cols]
    print("⚠️ Warning: Could not verify feature columns match exactly.")

# Generate submission
generate_submission(model, test_df, "data/submission.csv")
