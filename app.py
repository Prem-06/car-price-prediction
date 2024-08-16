from flask import Flask, request, jsonify
import joblib
import numpy as np
from waitress import serve

model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'price' not in data:
            return jsonify({'error': 'Price value is missing'}), 400
        
        price = np.array([[data['price']]])
        prediction = model.predict(price)
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8000)
