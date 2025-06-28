import pandas as pd
import joblib
from src.preprocessing import preprocess_data
from src.feature_engineering import add_features
from src.train_model import train_model
from utils import SafeLabelEncoder

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Rename for consistency
train.rename(columns={"Product_Rating": "Rating"}, inplace=True)

# Safe encoding
store_encoder = SafeLabelEncoder()
train['Store'] = store_encoder.fit_transform(train['Store'].astype(str))
test['Store'] = store_encoder.transform(test['Store'].astype(str))

# Save encoder
joblib.dump(store_encoder, "models/store_encoder.pkl")

# Preprocess + Feature Engineering
train = preprocess_data(train)
test = preprocess_data(test)
train = add_features(train)
test = add_features(test)

# Train
X = train.drop('Rating', axis=1)
y = train['Rating']
model = train_model(X, y)
joblib.dump(model, "models/final_model.pkl")

# Save test data for later use
test.to_csv("data/test_processed.csv", index=False)

print("✅ Model trained and test data prepared.")
