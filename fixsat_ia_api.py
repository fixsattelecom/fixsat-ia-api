
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
modelo = joblib.load('modelo_fixsat.pkl')
encoder = joblib.load('encoder_fixsat.pkl')

@app.route('/predict', methods=['POST'])
def prever():
    dados = request.get_json()
    entrada = np.array([[dados['p1'], dados['p2'], dados['p3'], dados['p4'], dados['p5'], dados['p14']]])
    pred = modelo.predict(entrada)
    defeito = encoder.inverse_transform(pred)[0]
    return jsonify({'defeito': defeito})

if __name__ == '__main__':
    app.run(debug=True)
