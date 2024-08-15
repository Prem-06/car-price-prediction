from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('model.pkl')
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        data = request.json
        price = data['price']
        price = np.array([[price]])
        prediction = model.predict(price)
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
