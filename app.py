from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib  # Для загрузки сохраненного масштабатора
import os

app = Flask(__name__)

# Путь к модели и масштабатору
model_path = os.path.join(os.path.dirname(__file__), 'model', 'lstm_model.keras')
scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'scaler.save')

# Загрузка обученной модели и масштабатора
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# Функция для предсказания цен
def predict_price(data):
    scaled_data = scaler.transform(data)
    scaled_data = scaled_data.reshape((1, len(data), 1))
    prediction = model.predict(scaled_data)
    return scaler.inverse_transform(prediction)[0, 0]

@app.route('/inference', methods=['POST'])
def inference():
    try:
        data = request.json.get('prices')
        if not data:
            return jsonify({"error": "No data provided"}), 400

        data = np.array(data).reshape(-1, 1)
        prediction = predict_price(data)
        return jsonify({"predicted_price": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
