from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('demographic_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')  # Load the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Get the age from the form
    age = request.form['age']

    # Convert the age to an array and scale it (as the model expects scaled input)
    input_data = np.array([[float(age)]])
    
    # Make prediction using the model
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    # Interpret the result
    result = 'Dementia' if prediction == 1 else 'No Dementia'
    confidence = prediction_proba[0] * 100

    return render_template('index.html', prediction_text=f"Prediction: {result}",
                           confidence_text=f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    app.run(debug=True)
