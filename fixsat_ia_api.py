
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

# Simular base de dados de treino (mesmo que original)
X_train = [
    [0, 620, 730, 525, 200, 0],   # Curto na malha VCC
    [570, 627, 730, 525, 0, 560], # Falha na realimentação
    [0, 627, 730, 525, 200, 560], # CI não habilita
    [572, 628, 731, 523, 200, 560], # OK
    [573, 621, 728, 528, 200, 562], # OK
    [0, 630, 720, 530, 150, 0], # Curto na malha VCC
    [580, 625, 735, 520, 0, 555], # Falha na realimentação
]

y_train_text = [
    "Curto na malha VCC",
    "Falha na realimentação",
    "CI não habilita",
    "OK",
    "OK",
    "Curto na malha VCC",
    "Falha na realimentação"
]

# Codificar as saídas
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train_text)

# Treinar o modelo diretamente
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def prever():
    dados = request.get_json()
    entrada = np.array([[dados['p1'], dados['p2'], dados['p3'], dados['p4'], dados['p5'], dados['p14']]])
    pred = modelo.predict(entrada)
    defeito = encoder.inverse_transform(pred)[0]
    return jsonify({'defeito': defeito})

if __name__ == '__main__':
    app.run(debug=True)
