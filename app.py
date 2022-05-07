from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello Tasmia"

@app.route('/predict',methods=['POST'])
def predict():
    hours = request.form.get('hours')

    input_query = np.array([[hours]]).reshape(-1,1)

    result = model.predict(input_query)
    predicted_result = result.tolist()
    #result = {'hours': hours}

    return jsonify({'score':predicted_result})

if __name__ == '__main__':
    app.run(debug=True)