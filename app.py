import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np


app=Flask(__name__)

# Loasd the ML Model

regmodel=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    new_data=np.array(list(data.values())).reshape(1,-1)
    output=regmodel.predict(new_data)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])

def predict():
    data=[float(x) for x in request.form.values()]
    new_data=np.array(data).reshape(1,-1)
    print(new_data)
    output=regmodel.predict(new_data)
    print(output)
    return render_template("home.html",predicted_value="The predicted value of the car is {} in lacs".format(output))

if __name__=="__main__":
    app.run(debug=True)