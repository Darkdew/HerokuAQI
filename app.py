#creating our web application

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

import pickle
app= Flask(__name__, template_folder='template')

file= open("LassoReg.pkl","rb")
model= pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)


    
    return render_template('index.html', prediction_text= 'AQI withthe given Weather Features should have been:{}'.format(output))

if __name__=="__main__":
    app.run(debug=True)