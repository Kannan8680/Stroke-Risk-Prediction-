import joblib
import numpy as np
import tensorflow as tf

# Load LSTM model and scaler
model = tf.keras.models.load_model("lstm_stroke_model.h5")
scaler = joblib.load("scaler_lstm.pkl")

def predict_stroke(features):
    """
    Predict stroke risk using the trained LSTM model.
    :param features: List of 4 values [Alpha, Beta, Theta, Delta]
    :return: 1 (High Stroke Risk) or 0 (Safe)
    """
    features_scaled = scaler.transform([features])
    features_reshaped = features_scaled.reshape((1, 1, len(features)))
    prediction = model.predict(features_reshaped)[0][0]
    
    return 1 if prediction >= 0.5 else 0  # Threshold at 0.5

# Example usage
example_data = [0.5, 0.7, 0.2, 0.6]  # Replace with real data
prediction = predict_stroke(example_data)

if prediction == 1:
    print("⚠️ Stroke Risk Detected!")
else:
    print("✅ You are Safe")

