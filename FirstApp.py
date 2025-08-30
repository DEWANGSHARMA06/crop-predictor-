from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load your trained model and label encoder
model = joblib.load('crop_recommender.pkl')
le = joblib.load('label_encoder.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Create a DataFrame with the input features, ensuring column names match training data
    input_df = pd.DataFrame([{
        'N': data['N'],
        'P': data['P'],
        'K': data['K'],
        'temperature': data['temperature'],
        'humidity': data['humidity'],
        'ph': data['ph'],
        'rainfall': data['rainfall']
    }])

    # Predict using the DataFrame input
    pred_encoded = model.predict(input_df)[0]
    pred_crop = le.inverse_transform([pred_encoded])[0]

    return jsonify({'predicted_crop': pred_crop})

if __name__ == '__main__':
    app.run(debug=True)
