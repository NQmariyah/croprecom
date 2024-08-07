from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('crop_prediction_model.joblib')

# Load the label encoder and scaler (if used during training)
label_encoder = joblib.load('label_encoder.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()

    # Convert data into a numpy array or the required format
    input_data = np.array([[data['N'], data['P'], data['K'], data['temperature'],
                            data['humidity'], data['pH'], data['rainfall']]])
    
    # Normalize the input data using the same scaler as during training
    input_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Convert numpy int32 to Python int
    prediction_label = label_encoder.inverse_transform(prediction)[0]
    
    # Return the result as JSON
    return jsonify({'prediction': str(prediction_label)})

if __name__ == '__main__':
    app.run(debug=True)
