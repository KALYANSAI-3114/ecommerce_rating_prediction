from flask import Flask, render_template, request
import pandas as pd
import joblib
from src.preprocessing import preprocess_data
from src.feature_engineering import add_features

app = Flask(__name__)

# Load model and encoder
model = joblib.load("models/final_model.pkl")
store_encoder = joblib.load("models/store_encoder.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    store = request.form['store']
    details = request.form['details']

    input_df = pd.DataFrame([{
        "Title": title,
        "Store": store,
        "Details": details
    }])

    # Encode using fitted encoder
    input_df['Store'] = store_encoder.transform(input_df['Store'].astype(str))

    # Preprocess + Feature Engineering
    input_df = preprocess_data(input_df)
    input_df = add_features(input_df)

    # Align features
    if hasattr(model, 'feature_names_in_'):
        input_df = input_df[model.feature_names_in_]

    rating = model.predict(input_df)[0]
    return render_template('result.html', rating=round(rating, 2))

if __name__ == '__main__':
    app.run(debug=True)
