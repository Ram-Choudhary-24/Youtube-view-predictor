from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the model and preprocessing tools
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract required numeric fields (example)
    try:
        input_data = {
            'comment_count': float(data.get('comments', 0)),
            'ratings_disabled': float(data.get('ratings', 0)),
            # Add more fields as needed — must match your `features`
        }

        # Convert to DataFrame with same column order
        input_df = pd.DataFrame([input_data], columns=features)

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        predicted_views = model.predict(input_scaled)[0]

        return jsonify({'predicted_views': int(predicted_views)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
